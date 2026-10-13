import argparse
import copy
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dynamic.engine import experiment_name
from src.utils import load_config


def main():
    p = argparse.ArgumentParser(description="Real-MD ablation matrix; dry-run does not generate results")
    p.add_argument("--config", default="dynamic_config.yaml")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--splits", nargs="+", choices=["random", "cold_protein", "cold_ligand", "cold_both"], default=["random"])
    p.add_argument("--temporal", nargs="+", choices=["gru", "mean", "first"], default=["gru"])
    p.add_argument("--epochs", type=int)
    p.add_argument("--include-topk", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    experiments = [("mlp", "classical", "l1"), ("mlp", "quantum", "l1"),
                   ("sae", "classical", "l1"), ("sae", "quantum", "l1"), ("dense_ae", "quantum", "l1")]
    if args.include_topk: experiments += [("sae", "classical", "topk"), ("sae", "quantum", "topk")]

    def run(command):
        if args.dry_run:
            import shlex
            print(shlex.join(command))
        else: subprocess.run(command, cwd=root, check=True)

    for split in args.splits:
        for temporal in args.temporal:
            for projector, scorer, mode in experiments:
                c = copy.deepcopy(config)
                c["model"].update(projector=projector, scorer=scorer)
                c["sae"]["sparsity_mode"] = mode
                c["dynamic"]["temporal_mode"] = temporal
                for seed in args.seeds:
                    command = [sys.executable, "train_dynamic.py", "--config", str(config_path), "--split", split,
                               "--seed", str(seed), "--projector", projector, "--model", scorer,
                               "--sparsity-mode", mode, "--temporal", temporal]
                    if args.manifest: command += ["--manifest", str(args.manifest.resolve())]
                    if args.epochs is not None: command += ["--epochs", str(args.epochs)]
                    run(command)
                run([sys.executable, "aggregate.py", "--root", str(Path(config["training"]["output"]) / experiment_name(c) / split)])


if __name__ == "__main__": main()
