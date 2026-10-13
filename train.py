from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.datasets import SPLITS, read_table, make_splits, make_loader
from src.engine import run_epoch, pin_loaded_revisions, table_fingerprint
from src.model import EncodedAffinityModel
from src.utils import load_config, seed_everything, choose_device, atomic_save, save_json


def main():
    p = argparse.ArgumentParser(description="Train affinity baseline without SAE")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--dataset", choices=["davis", "kiba", "bindingdb"], default="davis")
    p.add_argument("--model", choices=["classical", "quantum"], default="classical")
    p.add_argument("--split", choices=SPLITS, default="random")
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--output", type=Path, help="Unique run directory, must not already exist")
    p.add_argument("--device")
    args = p.parse_args()
    config = copy.deepcopy(load_config(args.config))
    if args.seed is not None:
        config["seed"] = args.seed
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.device:
        config["device"] = args.device
    seed_everything(config["seed"])
    device = choose_device(config["device"])
    t = config["training"]
    if t["epochs"] < 1 or t["patience"] < 1:
        raise ValueError("epochs/patience must be >=1")
    path = Path(config["data"]["root"]) / args.dataset / "pairs.csv"
    frame = read_table(path)
    splits = make_splits(frame, args.split, config["seed"], config["data"]["fractions"])
    run = args.output or Path(t["output"]) / args.dataset / args.model / args.split / f"seed_{config['seed']}"
    run.mkdir(parents=True, exist_ok=False)
    save_json({k: v.row_id.tolist() for k, v in splits.items()}, run / "splits.json")
    fingerprint = table_fingerprint(frame)
    mean = float(splits["train"].affinity.mean())
    std = max(float(splits["train"].affinity.std(ddof=0)), 1e-8)
    model = EncodedAffinityModel(config, args.model).to(device)
    # Keep analytic simulator CPU parameters CPU before optimizer creation.
    if args.model == "quantum":
        model.affinity.scoring_features.circuit.to("cpu")
    pin_loaded_revisions(config, model)
    (run / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    versions = subprocess.check_output([__import__('sys').executable, "-m", "pip", "freeze"], text=True)
    (run / "environment.txt").write_text(f"Python {platform.python_version()}\n{versions}")
    loaders = {k: make_loader(v, t["batch_size"], k == "train", config["seed"], t["num_workers"])
               for k, v in splits.items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=t["learning_rate"], weight_decay=t["weight_decay"], foreach=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5) if t["scheduler"] else None
    best, stale, history = float("inf"), 0, []
    print(f"Device={device}; trainable parameters={sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    for epoch in range(1, t["epochs"] + 1):
        train, _ = run_epoch(model, loaders["train"], device, mean, std, optimizer, t["gradient_clip"])
        val, _ = run_epoch(model, loaders["val"], device, mean, std)
        score = val["normalized_mse"]
        history.append({"epoch": epoch, "train_loss": train["normalized_mse"], "val_loss": score,
                        "val_rmse": val["rmse"], "lr": optimizer.param_groups[0]["lr"]})
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        print(json.dumps(history[-1]), flush=True)
        if scheduler:
            scheduler.step(score)
        if score < best - t["min_delta"]:
            best, stale = score, 0
            atomic_save({"state": model.checkpoint_state(), "config": config, "variant": args.model,
                         "dataset": args.dataset, "split": args.split, "seed": config["seed"],
                         "epoch": epoch, "mean": mean, "std": std,
                         "fingerprint": fingerprint, "data_path": str(path.resolve()),
                         "splits": {k: v.row_id.tolist() for k, v in splits.items()}}, run / "best.pt")
        else:
            stale += 1
        if stale >= t["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break
    checkpoint = torch.load(run / "best.pt", weights_only=True, map_location="cpu")
    model.restore(checkpoint["state"])
    test, predictions = run_epoch(model, loaders["test"], device, mean, std)
    save_json({"dataset": args.dataset, "model": args.model, "split": args.split,
               "seed": config["seed"], "best_epoch": checkpoint["epoch"], **test}, run / "metrics.json")
    pd.DataFrame(predictions).to_csv(run / "predictions.csv", index=False)
    print(f"Test: {test}\nCheckpoint: {run / 'best.pt'}")


if __name__ == "__main__":
    main()
