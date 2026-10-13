import argparse
from pathlib import Path

import pandas as pd
import torch

from src.dynamic.data import read_manifest, fingerprint, loader
from src.dynamic.model import DynamicAffinityModel
from src.dynamic.engine import run_epoch
from src.utils import choose_device, seed_everything, save_json


def main():
    p = argparse.ArgumentParser(description="Reload MD checkpoint and preserve its complex-level split")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--partition", choices=["train", "val", "test"], default="test")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    if checkpoint.get("kind") != "dynamic_affinity_v1": raise ValueError("Not an MD checkpoint")
    config = checkpoint["config"]
    seed_everything(config["seed"])
    frame = read_manifest(args.manifest or checkpoint["manifest"])
    if fingerprint(frame) != checkpoint["fingerprint"]: raise ValueError("MD manifest or trajectory contents have changed")
    selected = frame[frame.complex_id.isin(checkpoint["splits"][args.partition])]
    device = choose_device(args.device)
    model = DynamicAffinityModel(config).to(device)
    model.restore(checkpoint["state"])
    metrics, predictions = run_epoch(model, loader(selected, config), device, checkpoint["mean"], checkpoint["std"],
                                     export=args.checkpoint.parent / f"evaluation_{args.partition}.npz")
    save_json(metrics, args.checkpoint.parent / f"evaluation_{args.partition}.json")
    pd.DataFrame(predictions).to_csv(args.checkpoint.parent / f"evaluation_{args.partition}.csv", index=False)
    print(metrics)


if __name__ == "__main__": main()
