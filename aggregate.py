import argparse
import json
from pathlib import Path

import numpy as np

from src.utils import save_json


def main():
    p = argparse.ArgumentParser(description="Aggregate independent seeds, sample std (ddof=1)")
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    paths = sorted(args.root.glob("seed_*/metrics.json"))
    if not paths:
        raise ValueError("No seed_*/metrics.json found")
    runs = [json.loads(path.read_text()) for path in paths]
    if len({(r['dataset'], r['model'], r['split']) for r in runs}) != 1:
        raise ValueError("Do not mix datasets, variants or split policies")
    if len({r['seed'] for r in runs}) != len(runs):
        raise ValueError("Duplicate seeds")
    summary = {"seeds": [r["seed"] for r in runs], "std_definition": "sample, ddof=1"}
    for metric in ("rmse", "mae", "pearson", "spearman", "ci"):
        values = [r[metric] for r in runs if r[metric] is not None]
        summary[metric] = {"mean": float(np.mean(values)) if values else None,
                           "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                           "valid_seeds": len(values), "undefined_seeds": len(runs) - len(values)}
        print(f"{metric}: {summary[metric]}")
    save_json(summary, args.root / "summary.json")


if __name__ == "__main__":
    main()
