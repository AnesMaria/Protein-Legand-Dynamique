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
    group_keys = ("dataset", "model", "split", "projector", "sparsity_mode", "training_protocol",
                  "patch_encoder_frozen", "autoencoder_frozen", "experiment_signature", "dataset_fingerprint")
    if len({tuple(r.get(k) for k in group_keys) for r in runs}) != 1:
        raise ValueError("Do not mix datasets, projectors, sparsity modes, protocols or splits")
    if len({r['seed'] for r in runs}) != len(runs):
        raise ValueError("Duplicate seeds")
    summary = {"seeds": [r["seed"] for r in runs], "std_definition": "sample, ddof=1",
               "experiment": {k: runs[0].get(k) for k in group_keys}}
    for metric in ("rmse", "mae", "pearson", "spearman", "ci", "trainable_parameters", "training_seconds",
                   "inference_seconds", "reconstruction_mse", "latent_sparsity_ratio", "latent_fraction_zeros",
                   "latent_mean_active", "latent_mean_l1_norm", "latent_variance"):
        if not any(metric in r for r in runs):
            continue
        values = [r[metric] for r in runs if r.get(metric) is not None]
        summary[metric] = {"mean": float(np.mean(values)) if values else None,
                           "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                           "valid_seeds": len(values), "undefined_seeds": len(runs) - len(values)}
        print(f"{metric}: {summary[metric]}")
    save_json(summary, args.root / "summary.json")


if __name__ == "__main__":
    main()
