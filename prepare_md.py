import argparse
import json
from pathlib import Path

import pandas as pd

from src.dynamic.prepare import convert_misato
from src.dynamic.data import read_manifest, read_trajectory
from src.utils import save_json


def main():
    p = argparse.ArgumentParser(description="MISATO HDF5 -> real temporal trajectories and exact experimental labels")
    p.add_argument("--source", choices=["misato", "canonical"], default="misato")
    p.add_argument("--md-h5", type=Path)
    p.add_argument("--labels-h5", type=Path)
    p.add_argument("--manifest", type=Path, help="Validate canonical imports from MDbind/PLAS or other MD")
    p.add_argument("--endpoint", choices=["Kd", "Ki", "IC50"], default="Kd")
    p.add_argument("--output", type=Path, default=Path("data/dynamic/misato"))
    p.add_argument("--max-frames", type=int, default=16)
    p.add_argument("--max-protein-atoms", type=int, default=128)
    p.add_argument("--pocket-radius", type=float, default=8.)
    p.add_argument("--metadata", type=Path, help="JSON keyed by PDB ID, with true sequence/SMILES/group IDs")
    p.add_argument("--dt-ps", type=float, help="Only when physical snapshot interval is known")
    args = p.parse_args()
    if args.source == "canonical":
        if not args.manifest: p.error("--manifest required for canonical data")
        frame = read_manifest(args.manifest)
        for record in frame.to_dict("records"):
            for replica in record["trajectories"]: read_trajectory(replica["trajectory_path"])
        print(f"Validated {len(frame)} complexes; replicas/time/endpoint checked")
        return
    if args.md_h5 is None or args.labels_h5 is None:
        p.error("--md-h5 and --labels-h5 required; use scripts/fetch_misato_example.py for a small real sample")
    metadata = json.loads(args.metadata.read_text()) if args.metadata else None
    records, provenance = convert_misato(args.md_h5, args.labels_h5, args.output, args.endpoint,
                                         args.max_frames, args.max_protein_atoms, args.pocket_radius, metadata, args.dt_ps)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(args.output / "manifest.csv", index=False)
    save_json(provenance, args.output / "provenance.json")
    print(f"Prepared {len(records)} real MD complexes, target p{args.endpoint}; {args.output / 'manifest.csv'}")


if __name__ == "__main__": main()
