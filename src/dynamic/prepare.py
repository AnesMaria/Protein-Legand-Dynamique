"""Canonicalize MD coordinates without turning static conformers into fake trajectories."""
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from .data import validate_trajectory


def frame_selection(count, max_frames=16, stride=1):
    if count < 2 or max_frames < 2 or stride < 1:
        raise ValueError("Require genuine multi-frame data, max_frames>=2 and stride>=1")
    selected = np.arange(0, count, stride)
    if len(selected) > max_frames:
        selected = selected[np.linspace(0, len(selected) - 1, max_frames).round().astype(int)]
    if len(selected) < 2:
        raise ValueError("Selection leaves fewer than 2 frames")
    return selected


def crop_trajectory(positions, atomic_numbers, ligand_mask, max_protein_atoms=128, pocket_radius=8.):
    """Use one atom subset across time; rank protein atoms by their closest ligand approach."""
    if max_protein_atoms < 1 or pocket_radius <= 0:
        raise ValueError("Invalid atom budget/radius")
    protein = np.flatnonzero(~ligand_mask & (atomic_numbers != 1))
    ligand = np.flatnonzero(ligand_mask & (atomic_numbers != 1))
    if not len(protein) or not len(ligand):
        raise ValueError("No heavy protein or ligand atoms")
    closest = np.full(len(protein), np.inf)
    for frame in positions:
        closest = np.minimum(closest, cKDTree(frame[ligand]).query(frame[protein])[0])
    nearby = np.flatnonzero(closest <= pocket_radius)
    fallback = not len(nearby)
    if fallback:
        nearby = np.arange(len(protein))
    order = nearby[np.argsort(closest[nearby], kind="stable")[:max_protein_atoms]]
    selected = np.concatenate([np.sort(protein[order]), ligand])
    return selected, {"protein_heavy_before_crop": len(protein), "protein_atoms": len(order),
                       "ligand_atoms": len(ligand), "nearest_atom_fallback": fallback,
                       "truncated_pocket": len(nearby) > max_protein_atoms}


def save_canonical(path, positions, atomic_numbers, ligand_mask, frame_indices, time=None,
                   time_unit="frame_index", atom_ids=None):
    arrays = {"positions": np.asarray(positions, dtype=np.float32),
              "atomic_numbers": np.asarray(atomic_numbers, dtype=np.int64),
              "ligand_mask": np.asarray(ligand_mask, dtype=bool),
              "frame_indices": np.asarray(frame_indices, dtype=np.int64),
              "time": np.asarray(frame_indices if time is None else time, dtype=np.float64),
              "time_unit": np.array(time_unit), "units": np.array("angstrom"),
              "reference_positions": np.asarray(positions[0], dtype=np.float32),
              "atom_ids": np.asarray(np.arange(len(atomic_numbers)) if atom_ids is None else atom_ids, dtype=np.int64)}
    validate_trajectory(arrays)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def convert_misato(md_file, label_file, output, endpoint="Kd", max_frames=16, max_protein_atoms=128,
                   pocket_radius=8., metadata=None, dt_ps=None):
    output = Path(output)
    records, exclusions, crops = [], {}, {}
    metadata = metadata or {}
    if endpoint not in ("Kd", "Ki", "IC50") or dt_ps is not None and dt_ps <= 0:
        raise ValueError("Invalid endpoint/time interval")
    with h5py.File(md_file, "r") as md, h5py.File(label_file, "r") as labels:
        for complex_id in sorted(md.keys()):
            if complex_id not in labels:
                exclusions[complex_id] = "no experimental label"; continue
            label = labels[complex_id]
            kind = label["affinityType"][()].decode()
            relation = label["type"][()].decode()
            value = float(label["affinity"][()])
            if kind != endpoint + " (nM)" or relation != "=" or not np.isfinite(value) or value <= 0:
                exclusions[complex_id] = f"excluded endpoint/censoring/value: {kind}, {relation}"; continue
            group = md[complex_id]
            z = group["atoms_number"][:]  # Verified: atomic numbers; atoms_element contains mapped codes.
            boundaries = group["molecules_begin_atom_index"][:]
            if not len(boundaries) or boundaries[-1] <= 0 or boundaries[-1] >= len(z):
                raise ValueError(f"Unexpected molecule boundaries in {complex_id}")
            ligand = np.arange(len(z)) >= boundaries[-1]  # Last molecule, per MISATO convention.
            frames = frame_selection(len(group["trajectory_coordinates"]), max_frames)
            positions = group["trajectory_coordinates"][frames].astype(np.float32)
            selected, info = crop_trajectory(positions, z, ligand, max_protein_atoms, pocket_radius)
            crops[complex_id] = info
            path = output / "trajectories" / f"{complex_id}_r0.npz"
            save_canonical(path, positions[:, selected], z[selected], ligand[selected], frames,
                           time=frames * dt_ps if dt_ps is not None else None,
                           time_unit="ps" if dt_ps is not None else "frame_index", atom_ids=selected)
            extra = metadata.get(complex_id, {})
            protein_id = label["Uniprot"][()].decode()
            ligand_id = label["ligandID"][()].decode()
            # Generic peptide names such as '4-mer' are not verified chemical identities.
            records.append({"complex_id": complex_id, "trajectory_id": complex_id + "_r0", "replica_id": 0,
                            "protein_id": protein_id or complex_id, "ligand_id": ligand_id or complex_id,
                            "protein_group": extra.get("protein_group", protein_id),
                            "ligand_group": extra.get("ligand_group", ""),
                            "sequence": extra.get("sequence", ""), "smiles": extra.get("smiles", ""),
                            "trajectory_path": str(path.resolve()), "affinity": 9 - np.log10(value),
                            "target_type": "p" + endpoint, "source": "MISATO", "synthetic": False})
    if not records:
        raise ValueError("No exact positive labels match the chosen endpoint")
    return records, {"source_md": str(Path(md_file).resolve()), "source_labels": str(Path(label_file).resolve()),
                     "endpoint": endpoint, "complexes": len(records), "exclusions": exclusions, "crops": crops,
                     "time_unit": "ps" if dt_ps is not None else "frame_index", "dt_ps_user_supplied": dt_ps,
                     "coordinates": "MISATO coordinates as provided; no additional PBC correction inferred",
                     "crop_policy": "fixed heavy-atom subset from union of contacts over selected frames"}
