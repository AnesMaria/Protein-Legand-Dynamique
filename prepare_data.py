"""Convert checked upstream tables into a stable, auditable local schema."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import digest, save_json

POCKETDTA_URL = "https://github.com/BioCenter-SHU/PocketDTA.git"
POCKETDTA_COMMIT = "0e03b353544129fac537321787a1b38aba6e0192"
BALM_REVISION = "84e07e27a093181ee89c97a0d7a45482803aa9bd"


def ensure_pocketdta(path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", POCKETDTA_URL, str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--detach", POCKETDTA_COMMIT], check=True)
    if not (path / "data").exists():
        raise ValueError(f"Not a PocketDTA checkout: {path}")
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    return path, commit


def normalize(frame, dataset, graph_map=None, columns=None, affinity_unit="native"):
    columns = columns or {"protein_id": "Target_ID", "ligand_id": "Drug_ID",
                          "sequence": "Target", "smiles": "Drug", "affinity": "Y"}
    missing = set(columns.values()) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing source columns {missing}; use --columns mapping.json")
    out = frame[list(columns.values())].rename(columns={v: k for k, v in columns.items()}).copy()
    before = len(out)
    # Reject malformed data instead of silently changing benchmark populations.
    if out.isna().any().any():
        raise ValueError("Null source fields: inspect/filter upstream explicitly")
    out["sequence"] = out.sequence.astype(str).str.replace(r"\s+", "", regex=True).str.upper()
    out["smiles"] = out.smiles.astype(str).str.strip()
    if out.sequence.eq("").any() or out.smiles.eq("").any():
        raise ValueError("Empty sequence or SMILES")
    for key in ("protein_id", "ligand_id"):
        out[key] = out[key].astype(str)
    out["affinity"] = pd.to_numeric(out.affinity, errors="raise")
    if not np.isfinite(out.affinity).all():
        raise ValueError("Nonfinite affinities")
    if affinity_unit == "nM":
        if dataset == "kiba" or (out.affinity <= 0).any():
            raise ValueError("nM conversion requires positive Kd, not KIBA scores")
        out["affinity"] = 9 - np.log10(out.affinity)
    # BALM uses max Y for repeated pairs. Apply by sequence/SMILES too, avoiding aliases.
    reducer = "max" if dataset == "bindingdb" else "mean"
    out = out.groupby(["sequence", "smiles"], sort=True, as_index=False).agg(
        protein_id=("protein_id", "first"), ligand_id=("ligand_id", "first"),
        affinity=("affinity", reducer))
    mapping = graph_map or {}
    out["graph_path"] = out.protein_id.map(lambda k: mapping.get(str(k), ""))
    print(f"Normalized {before} rows -> {len(out)} unique sequence/SMILES pairs ({reducer})")
    return out, {"input_rows": before, "output_rows": len(out), "duplicate_reducer": reducer,
                 "affinity_unit": "KIBA score" if dataset == "kiba" else "pKd"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["davis", "kiba", "bindingdb"], required=True)
    p.add_argument("--source", type=Path, help="Local PocketDTA clone; else clone pinned source")
    p.add_argument("--input", type=Path, help="Local upstream CSV (alternative to automatic retrieval)")
    p.add_argument("--columns", type=Path, help='JSON: canonical field -> input column')
    p.add_argument("--graph-map", type=Path, help="JSON: protein_id -> tensor graph path")
    p.add_argument("--sequence-only", action="store_true", help="Skip automatic PDB/P2Rank adaptation")
    p.add_argument("--pocket-rank", type=int, default=1, help="P2Rank rank (0=all predicted pockets)")
    p.add_argument("--affinity-unit", choices=["native", "nM"], default="native")
    p.add_argument("--balm-revision", default=BALM_REVISION)
    p.add_argument("--output-root", type=Path, default=Path("data/processed"))
    args = p.parse_args()
    metadata = {"dataset": args.dataset}
    source = None
    if args.input:
        frame = pd.read_csv(args.input)
        metadata.update(source=str(args.input.resolve()), sha256=hashlib.sha256(args.input.read_bytes()).hexdigest())
    elif args.dataset == "bindingdb":
        from datasets import load_dataset
        from huggingface_hub import HfApi
        revision = HfApi().dataset_info("BALM/BALM-benchmark", revision=args.balm_revision).sha
        frame = load_dataset("BALM/BALM-benchmark", "BindingDB_filtered", split="train",
                             revision=revision).to_pandas()
        metadata.update(source="BALM/BALM-benchmark:BindingDB_filtered", commit=revision)
    else:
        source, commit = ensure_pocketdta(args.source or Path("data/raw/PocketDTA"))
        folder = "Davis" if args.dataset == "davis" else "KIBA"
        frame = pd.read_csv(source / "data" / folder / f"{args.dataset}_datasets.csv")
        metadata.update(source=POCKETDTA_URL, commit=commit)
    columns = json.loads(args.columns.read_text()) if args.columns else None
    graph_map = json.loads(args.graph_map.read_text()) if args.graph_map else {}
    if args.graph_map:
        graph_map = {k: str((args.graph_map.parent / v).resolve()) for k, v in graph_map.items()}
    elif source is not None and not args.sequence_only:
        from src.pocket_data import build_pocket_graphs
        graph_map = build_pocket_graphs(frame, source, args.output_root / args.dataset / "graphs",
                                       args.dataset, args.pocket_rank)
    out, stats = normalize(frame, args.dataset, graph_map, columns, args.affinity_unit)
    root = args.output_root / args.dataset
    root.mkdir(parents=True, exist_ok=True)
    out.to_csv(root / "pairs.csv", index=False)
    metadata.update(stats)
    metadata.update(graph_count=int(out.graph_path.ne("").sum()),
                    graph_fallback="sequence windows when no tensor graph supplied",
                    split_policy="local reproducible splits, not published PocketDTA/BALM folds")
    save_json(metadata, root / "provenance.json")
    print(f"Saved {root / 'pairs.csv'}; graphs={metadata['graph_count']}")


if __name__ == "__main__":
    main()
