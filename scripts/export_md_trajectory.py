"""Import native MD trajectories via MDAnalysis with explicit topology, selections and provenance."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.dynamic.prepare import frame_selection, crop_trajectory, save_canonical
from src.dynamic.data import read_manifest
from src.utils import save_json


def export(topology, trajectory, output, protein_selection, ligand_selection, max_frames=16,
           max_protein_atoms=128, physical_time=False, unwrap=False, pocket_radius=8.):
    import MDAnalysis as mda
    from MDAnalysis.topology.tables import SYMB2Z
    from MDAnalysis.lib.distances import minimize_vectors

    if Path(trajectory).suffix.lower() in (".pdb", ".mol2", ".sdf", ".gro"):
        raise ValueError("A static structure/conformer collection is not accepted as an MD trajectory")
    universe = mda.Universe(str(topology), str(trajectory))
    protein = universe.select_atoms(protein_selection)
    ligand = universe.select_atoms(ligand_selection)
    if not len(protein) or not len(ligand) or np.intersect1d(protein.indices, ligand.indices).size:
        raise ValueError("Require nonempty disjoint protein and ligand atom selections")
    combined = protein + ligand
    try:
        symbols = [str(e).upper() for e in combined.elements]
        atomic_numbers = np.array([SYMB2Z[e] for e in symbols], dtype=np.int64)
    except (AttributeError, KeyError, mda.exceptions.NoDataError) as exc:
        raise ValueError("Topology requires explicit valid elements; atom-name guessing is not used") from exc
    ligand_mask = np.arange(len(combined)) >= len(protein)
    frames = frame_selection(len(universe.trajectory), max_frames)
    coordinates, times = [], []
    for frame_index in frames:
        ts = universe.trajectory[int(frame_index)]
        positions = combined.positions.copy()
        if unwrap:
            if ts.dimensions is None or not np.all(np.isfinite(ts.dimensions)):
                raise ValueError("PBC unwrapping requires valid unit-cell dimensions")
            # Make every bonded fragment whole; topology must provide bonds.
            combined.unwrap(compound="fragments", reference=None, inplace=True)
            positions = combined.positions.copy()
            ligand_center = positions[ligand_mask].mean(0)
            vectors = minimize_vectors(ligand_center - positions[~ligand_mask], ts.dimensions)
            closest = np.linalg.norm(vectors, axis=-1).argmin()
            shift = positions[~ligand_mask][closest] + vectors[closest] - ligand_center
            positions[ligand_mask] += shift
        coordinates.append(positions)
        times.append(float(ts.time) if physical_time else int(frame_index))
    coordinates = np.stack(coordinates)
    selected, crop = crop_trajectory(coordinates, atomic_numbers, ligand_mask, max_protein_atoms, pocket_radius)
    save_canonical(output, coordinates[:, selected], atomic_numbers[selected], ligand_mask[selected], frames,
                   time=times, time_unit="ps" if physical_time else "frame_index", atom_ids=combined.indices[selected])
    return {"topology": str(Path(topology).resolve()), "trajectory": str(Path(trajectory).resolve()),
            "protein_selection": protein_selection, "ligand_selection": ligand_selection,
            "time": "MDAnalysis reader times in ps, explicitly requested" if physical_time else "source frame indices; physical dt not asserted",
            "pbc": "bonded-fragment unwrap and ligand minimum-image shift" if unwrap else "as provided; no PBC correction",
            "crop": crop, "selected_frames": frames.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="CSV of topology/trajectory paths, IDs and experimental labels")
    p.add_argument("--dataset", choices=["mdbind", "plas20k", "custom"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protein-selection", default="protein")
    p.add_argument("--ligand-selection", required=True, help="E.g. resname LIG; never select solvent")
    p.add_argument("--max-frames", type=int, default=16)
    p.add_argument("--max-protein-atoms", type=int, default=128)
    p.add_argument("--physical-time", action="store_true", help="Only when reader header dt/time is known to be physical")
    p.add_argument("--unwrap", action="store_true", help="Requires topology bonds and box; see README limitations for multicomponent systems")
    args = p.parse_args()
    frame = pd.read_csv(args.input, keep_default_na=False)
    required = {"complex_id", "trajectory_id", "protein_id", "ligand_id", "topology", "trajectory", "affinity", "target_type"}
    if not required.issubset(frame): p.error(f"Missing input columns {required - set(frame)}")
    records, provenance = [], {}
    for index, record in enumerate(frame.to_dict("records")):
        path = args.output / "trajectories" / f"replica_{index:06d}.npz"
        topology = args.input.parent / record.pop("topology")
        trajectory = args.input.parent / record.pop("trajectory")
        provenance[record["trajectory_id"]] = export(topology, trajectory, path, args.protein_selection,
                                                    args.ligand_selection, args.max_frames, args.max_protein_atoms,
                                                    args.physical_time, args.unwrap)
        record.update(trajectory_path=str(path.resolve()), source=args.dataset, synthetic=False)
        records.append(record)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output / "manifest.csv", index=False)
    read_manifest(args.output / "manifest.csv")
    save_json(provenance, args.output / "provenance.json")
    print(args.output / "manifest.csv")


if __name__ == "__main__": main()
