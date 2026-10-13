import argparse
from pathlib import Path

import pandas as pd
import torch

from src.datasets import read_table, make_loader
from src.engine import table_fingerprint
from src.train_utils import run_epoch
from src.model import EncodedAffinityModel
from src.utils import choose_device, seed_everything, save_json


def main():
    p = argparse.ArgumentParser(description="Evaluate saved split and best checkpoint")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data", type=Path, help="Override path to the same prepared CSV")
    p.add_argument("--partition", choices=["train", "val", "test"], default="test")
    p.add_argument("--device", default="auto")
    p.add_argument("--save-latents", type=Path, help="Export valid patch latents and pair IDs as NPZ")
    args = p.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    seed_everything(checkpoint["seed"])
    device = choose_device(args.device)
    frame = read_table(args.data or checkpoint["data_path"])
    if table_fingerprint(frame) != checkpoint["fingerprint"]:
        raise ValueError("Data differ from training; saved row split cannot be reused safely")
    frame = frame.iloc[checkpoint["splits"][args.partition]]
    model = EncodedAffinityModel(config, checkpoint["variant"]).to(device)
    model.restore(checkpoint["state"])
    loader = make_loader(frame, config["training"]["batch_size"], False, checkpoint["seed"])
    metrics, predictions = run_epoch(model, loader, device, checkpoint["mean"], checkpoint["std"],
                                     latent_path=args.save_latents)
    save_json(metrics, args.checkpoint.parent / f"evaluation_{args.partition}.json")
    pd.DataFrame(predictions).to_csv(args.checkpoint.parent / f"evaluation_{args.partition}.csv", index=False)
    print(metrics)


if __name__ == "__main__":
    main()
