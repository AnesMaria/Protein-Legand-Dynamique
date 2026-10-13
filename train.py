from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
from time import perf_counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from src.datasets import SPLITS, read_table, make_splits, make_loader
from src.engine import pin_loaded_revisions, table_fingerprint
from src.train_utils import run_epoch, apply_pretraining, default_pretrain_path, projector_label, experiment_signature
from src.model import EncodedAffinityModel
from src.utils import load_config, seed_everything, choose_device, atomic_save, save_json


def main():
    p = argparse.ArgumentParser(description="Compare MLP, sparse AE and dense AE affinity models")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--dataset", choices=["davis", "kiba", "bindingdb"], default="davis")
    p.add_argument("--model", choices=["classical", "quantum"])
    p.add_argument("--projector", choices=["mlp", "sae", "dense_ae"])
    p.add_argument("--sparsity-mode", choices=["l1", "topk"])
    p.add_argument("--sae-checkpoint", type=Path)
    p.add_argument("--patch-checkpoint", type=Path, help="Shared patch representation for controlled ablations")
    p.add_argument("--no-pretrained", action="store_true", help="Explicit end-to-end mode")
    p.add_argument("--freeze-sae", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--freeze-patch-encoder", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--split", choices=SPLITS, default="random")
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--output", type=Path, help="Unique run directory, must not already exist")
    p.add_argument("--device")
    args = p.parse_args()
    config = copy.deepcopy(load_config(args.config))
    args.model = args.model or config["model"].get("scorer", "classical")
    config["model"]["scorer"] = args.model
    config["model"]["projector"] = args.projector or config["model"].get("projector", "mlp")
    if args.sparsity_mode:
        config["sae"]["sparsity_mode"] = args.sparsity_mode
    if args.freeze_sae is not None:
        config["sae"]["freeze_after_pretrain"] = args.freeze_sae
    if args.freeze_patch_encoder is not None:
        config["model"]["freeze_patch_encoder"] = args.freeze_patch_encoder
    if args.no_pretrained and (args.sae_checkpoint or args.patch_checkpoint):
        p.error("--no-pretrained cannot be combined with checkpoint transfers")
    if args.sae_checkpoint and config["model"]["projector"] == "mlp":
        p.error("Select --projector sae or dense_ae to load an autoencoder")
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
    run = args.output or Path(t["output"]) / args.dataset / projector_label(config) / args.model / args.split / f"seed_{config['seed']}"
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
    context = {"dataset": args.dataset, "split": args.split, "seed": config["seed"],
               "fingerprint": fingerprint, "training_rows": splits["train"].row_id.tolist()}
    sae_path = None
    if not args.no_pretrained and config["model"]["projector"] != "mlp":
        supplied = args.sae_checkpoint or config["sae"].get("pretrained_checkpoint")
        default = default_pretrain_path(config, args.dataset, args.split, config["seed"])
        sae_path = supplied or (default if default.exists() else None)
    protocol = apply_pretraining(model, config, context, sae_path, args.patch_checkpoint)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    (run / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    versions = subprocess.check_output([__import__('sys').executable, "-m", "pip", "freeze"], text=True)
    (run / "environment.txt").write_text(f"Python {platform.python_version()}\n{versions}")
    loaders = {k: make_loader(v, t["batch_size"], k == "train", config["seed"], t["num_workers"])
               for k, v in splits.items()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=t["learning_rate"], weight_decay=t["weight_decay"], foreach=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5) if t["scheduler"] else None
    best, stale, history = float("inf"), 0, []
    print(f"Device={device}; projector={projector_label(config)}; protocol={protocol}; trainable parameters={trainable}")
    started = perf_counter()
    for epoch in range(1, t["epochs"] + 1):
        train, _ = run_epoch(model, loaders["train"], device, mean, std, optimizer, t["gradient_clip"])
        val, _ = run_epoch(model, loaders["val"], device, mean, std)
        score = val["normalized_mse"]
        history.append({"epoch": epoch, "train_loss": train["normalized_mse"], "val_loss": score,
                        "val_rmse": val["rmse"], "lr": optimizer.param_groups[0]["lr"]})
        for name, metrics in (("train", train), ("val", val)):
            history[-1].update({name + "_" + k: v for k, v in metrics.items() if k not in ("n", "normalized_mse")})
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        print(json.dumps(history[-1]), flush=True)
        if scheduler:
            scheduler.step(score)
        if score < best - t["min_delta"]:
            best, stale = score, 0
            atomic_save({"state": model.checkpoint_state(), "config": config, "variant": args.model,
                         "dataset": args.dataset, "split": args.split, "seed": config["seed"],
                         "epoch": epoch, "mean": mean, "std": std,
                         "projector": config["model"]["projector"], "training_protocol": protocol,
                         "trainable_parameters": trainable,
                         "fingerprint": fingerprint, "data_path": str(path.resolve()),
                         "splits": {k: v.row_id.tolist() for k, v in splits.items()}}, run / "best.pt")
        else:
            stale += 1
        if stale >= t["patience"]:
            print(f"Early stopping at epoch {epoch}")
            break
    training_seconds = perf_counter() - started
    checkpoint = torch.load(run / "best.pt", weights_only=True, map_location="cpu")
    model.restore(checkpoint["state"])
    latent_path = run / "latents_test.npz" if t.get("save_latents", True) else None
    test, predictions = run_epoch(model, loaders["test"], device, mean, std, latent_path=latent_path)
    save_json({"dataset": args.dataset, "model": args.model, "split": args.split,
               "seed": config["seed"], "best_epoch": checkpoint["epoch"], **test}, run / "metrics.json")
    result = json.loads((run / "metrics.json").read_text())
    result.update(projector=config["model"]["projector"], experiment=projector_label(config),
                  sparsity_mode=config.get("sae", {}).get("sparsity_mode") if config["model"]["projector"] == "sae" else "none",
                  training_protocol=protocol, trainable_parameters=trainable, training_seconds=training_seconds,
                  inference_seconds=test["elapsed_seconds"], patch_encoder_frozen=config["model"].get("freeze_patch_encoder", False),
                  autoencoder_frozen=config.get("sae", {}).get("frozen", False),
                  experiment_signature=experiment_signature(config), dataset_fingerprint=fingerprint)
    save_json(result, run / "metrics.json")
    pd.DataFrame(predictions).to_csv(run / "predictions.csv", index=False)
    print(f"Test: {test}\nCheckpoint: {run / 'best.pt'}")


if __name__ == "__main__":
    main()
