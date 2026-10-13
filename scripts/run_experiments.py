"""Launch controlled ablations sequentially; never invent results for unrun cells."""
import argparse
import copy
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets import SPLITS
from src.train_utils import default_pretrain_path
from src.utils import load_config

EXPERIMENTS = [("mlp", "classical", None), ("mlp", "quantum", None),
               ("sae", "classical", "l1"), ("sae", "quantum", "l1"),
               ("dense_ae", "quantum", None)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--datasets", nargs="+", choices=["davis", "kiba", "bindingdb"], default=["davis"])
    p.add_argument("--splits", nargs="+", choices=SPLITS, default=["random"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--epochs", type=int)
    p.add_argument("--pretrain-epochs", type=int)
    p.add_argument("--include-topk", action="store_true")
    p.add_argument("--protocol", choices=["end_to_end", "pretrained"], default="end_to_end")
    p.add_argument("--dry-run", action="store_true", help="Print commands, no training or results")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    experiments = EXPERIMENTS + ([("sae", "classical", "topk"), ("sae", "quantum", "topk")] if args.include_topk else [])

    def run(arguments):
        command = [sys.executable, *arguments]
        if args.dry_run:
            import shlex
            print(shlex.join(command))
        else:
            subprocess.run(command, cwd=root, check=True)

    for dataset in args.datasets:
        for split in args.splits:
            for seed in args.seeds:
                common = ["--dataset", dataset, "--split", split, "--seed", str(seed), "--config", str(config_path)]
                pretrained = {}
                if args.protocol == "pretrained":
                    for projector, mode in [("sae", "l1"), ("dense_ae", "none")] + ([("sae", "topk")] if args.include_topk else []):
                        c = copy.deepcopy(config)
                        c["model"]["projector"] = projector
                        if mode != "none":
                            c["sae"]["sparsity_mode"] = mode
                        path = default_pretrain_path(c, dataset, split, seed)
                        pretrained[(projector, mode)] = path
                        command = ["pretrain_sae.py", *common, "--projector", projector]
                        if mode != "none":
                            command += ["--sparsity-mode", mode]
                        if args.pretrain_epochs is not None:
                            command += ["--epochs", str(args.pretrain_epochs)]
                        run(command)
                for projector, scorer, mode in experiments:
                    command = ["train.py", *common, "--projector", projector, "--model", scorer]
                    if mode is not None:
                        command += ["--sparsity-mode", mode]
                    if args.epochs is not None:
                        command += ["--epochs", str(args.epochs)]
                    if args.protocol == "end_to_end":
                        command += ["--no-pretrained", "--no-freeze-sae"]
                    else:
                        # Identical seeded, frozen patch representations in every pretrained ablation.
                        command += ["--freeze-patch-encoder"]
                        if projector == "mlp":
                            command += ["--patch-checkpoint", str(pretrained[("sae", "l1")])]
                        else:
                            command += ["--sae-checkpoint", str(pretrained[(projector, mode or "none")])]
                    run(command)
            for projector, scorer, mode in experiments:
                label = f"sae_{mode}" if projector == "sae" else projector
                output = Path(config["training"]["output"]) / dataset / label / scorer / split
                run(["aggregate.py", "--root", str(output)])


if __name__ == "__main__":
    main()
