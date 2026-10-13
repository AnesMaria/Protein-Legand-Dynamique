"""Summarize exported latents without fitting any transform on held-out data."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.utils import save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, help="JSON diagnostics (default: next to NPZ)")
    p.add_argument("--threshold", type=float, help="Default: threshold saved with the latents")
    p.add_argument("--bins", type=int, default=40)
    args = p.parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        z = archive["z"].astype(np.float64)
        threshold = args.threshold if args.threshold is not None else float(archive["activity_threshold"])
        rows = archive["row_id"] if "row_id" in archive else None
    if z.ndim != 2 or not len(z) or not np.isfinite(z).all() or threshold < 0 or args.bins < 1:
        raise ValueError("Invalid latent matrix, threshold or bins")
    active = np.abs(z) > threshold
    counts, edges = np.histogram(z, bins=args.bins)
    unique = len(np.unique(rows)) if rows is not None else None
    per_latent_activity = active.mean(0)
    report = {"patches": len(z), "latent_dim": z.shape[1], "pairs": unique,
              "activity_threshold": threshold, "active_latent_ratio": float(active.mean()),
              "sparsity_ratio": float(1 - active.mean()), "fraction_zeros": float((z == 0).mean()),
              "mean_active_latents": float(active.sum(1).mean()),
              "mean_l1_norm": float(np.abs(z).sum(1).mean()),
              "mean_latent_variance": float(z.var(0).mean()),
              "dead_latent_fraction": float((per_latent_activity == 0).mean()),
              "per_latent_active_ratio": per_latent_activity.tolist(),
              "per_latent_variance": z.var(0).tolist(),
              "activation_histogram": {"counts": counts.tolist(), "edges": edges.tolist()},
              "weighting": "per valid patch; the same protein may recur for multiple ligand pairs"}
    output = args.output or args.input.with_suffix(".analysis.json")
    save_json(report, output)
    print(f"{output}: patches={len(z)}, activity={active.mean():.4f}, dead={report['dead_latent_fraction']:.4f}")


if __name__ == "__main__":
    main()
