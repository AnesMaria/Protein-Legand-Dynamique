"""Pretrain only on patches from the affinity training partition, without labels."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.datasets import SPLITS, read_table, make_splits
from src.engine import pin_loaded_revisions, table_fingerprint
from src.model import EncodedAffinityModel
from src.sae_losses import reconstruction_loss, sparsity_loss
from src.train_utils import LatentStats, default_pretrain_path, feature_signature, load_patch_reference
from src.utils import load_config, seed_everything, choose_device, atomic_save, save_json, digest


def fit_epoch(autoencoder, patches, batch_size, optimizer, generator, threshold, device):
    training = optimizer is not None
    autoencoder.train(training)
    loader = DataLoader(TensorDataset(patches), batch_size=batch_size, shuffle=training, generator=generator)
    stats = LatentStats(threshold, autoencoder.topk if autoencoder.sparsity_mode == "topk" else None)
    for (x,) in loader:
        x = x.to(device)
        with torch.set_grad_enabled(training):
            z, reconstructed = autoencoder(x)
            reconstruction = reconstruction_loss(reconstructed, x)
            sparse = sparsity_loss(z, autoencoder.sparsity_mode)
            loss = reconstruction + autoencoder.lambda_sparse * sparse
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite pretraining loss")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(autoencoder.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
        stats.update({"latent": z}, reconstruction, sparse)
    metrics = stats.metrics()
    metrics["total_loss"] = metrics["reconstruction_mse"] + autoencoder.lambda_sparse * metrics["sparsity_loss"]
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--dataset", choices=["davis", "kiba", "bindingdb"], default="davis")
    p.add_argument("--split", choices=SPLITS, default="random")
    p.add_argument("--projector", choices=["sae", "dense_ae"], default="sae")
    p.add_argument("--sparsity-mode", choices=["l1", "topk"])
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--output", type=Path, help="Unique pretraining directory")
    p.add_argument("--patch-checkpoint", type=Path, help="Optional compatible trained baseline patch encoder")
    p.add_argument("--limit-proteins", type=int, help="Explicit debug-only cap")
    p.add_argument("--device")
    args = p.parse_args()
    config = copy.deepcopy(load_config(args.config))
    config["model"]["projector"] = args.projector
    config["sae"]["frozen"] = False
    if args.sparsity_mode:
        config["sae"]["sparsity_mode"] = args.sparsity_mode
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device:
        config["device"] = args.device
    s = config["sae"]
    epochs = args.epochs if args.epochs is not None else s["pretrain_epochs"]
    if epochs < 1 or s["lambda_sparse"] < 0 or not 0 < s["pretrain_validation_fraction"] < 1:
        raise ValueError("Invalid epochs, lambda_sparse or inner validation fraction")
    if not config["encoders"]["protein"]["freeze"]:
        raise ValueError("Patch-only pretraining requires the ESM backbone to be frozen")
    seed_everything(config["seed"])
    device = choose_device(config["device"])
    frame = read_table(Path(config["data"]["root"]) / args.dataset / "pairs.csv")
    splits = make_splits(frame, args.split, config["seed"], config["data"]["fractions"])
    context = {"dataset": args.dataset, "split": args.split, "seed": config["seed"],
               "fingerprint": table_fingerprint(frame), "training_rows": splits["train"].row_id.tolist()}
    run = args.output or default_pretrain_path(config, args.dataset, args.split, config["seed"]).parent
    run.mkdir(parents=True, exist_ok=False)
    # Use the classical scorer to avoid an unused quantum simulator; only the AE is optimized.
    model = EncodedAffinityModel(config, "classical").to(device).eval()
    pin_loaded_revisions(config, model)
    if args.patch_checkpoint:
        load_patch_reference(model, args.patch_checkpoint, config, context)
    model.affinity.patches.requires_grad_(False)
    model.protein_encoder.requires_grad_(False)
    records = splits["train"].drop_duplicates(["protein_key", "graph_path"]).to_dict("records")
    if args.limit_proteins is not None:
        if args.limit_proteins < 2:
            raise ValueError("Pretraining debug cap must retain at least 2 proteins")
        records = records[:args.limit_proteins]
    if len(records) < 2:
        raise ValueError("At least 2 training protein/graph entries required for inner validation")
    # Inner validation is disjoint by protein sequence, and drawn only from affinity-train.
    proteins = sorted(set(r["protein_key"] for r in records))
    if len(proteins) < 2:
        raise ValueError("Need at least 2 distinct training protein sequences")
    rng = np.random.default_rng(config["seed"])
    proteins = rng.permutation(proteins).tolist()
    n_val = min(len(proteins) - 1, max(1, int(len(proteins) * s["pretrain_validation_fraction"])))
    held_out = set(proteins[:n_val])
    groups = {"train": [], "val": []}
    with torch.no_grad():
        for index, record in enumerate(records):
            patch = model.encode_patch_records([record])[0].detach().cpu()
            groups["val" if record["protein_key"] in held_out else "train"].append(patch)
            if index == 0 or (index + 1) % 25 == 0 or index + 1 == len(records):
                print(f"Patches {index + 1}/{len(records)}: {tuple(patch.shape)}", flush=True)
    matrices = {k: torch.cat(v) for k, v in groups.items()}
    atomic_save({**matrices, "train_proteins": [k for k in proteins if k not in held_out],
                 "val_proteins": sorted(held_out)}, run / "patches.pt")
    autoencoder = model.affinity.projection.autoencoder
    # This coefficient is not a parameter; it documents the pretraining objective.
    autoencoder.lambda_sparse = s["lambda_sparse"] if args.projector == "sae" else 0.
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=s["pretrain_learning_rate"],
                                 weight_decay=config["training"]["weight_decay"])
    generator = torch.Generator().manual_seed(config["seed"])
    best, stale, history = float("inf"), 0, []
    started = perf_counter()
    for epoch in range(1, epochs + 1):
        train = fit_epoch(autoencoder, matrices["train"], s["pretrain_batch_size"], optimizer,
                          generator, s["activity_threshold"], device)
        val = fit_epoch(autoencoder, matrices["val"], s["pretrain_batch_size"], None,
                        generator, s["activity_threshold"], device)
        history.append({"epoch": epoch, **{"train_" + k: v for k, v in train.items()},
                        **{"val_" + k: v for k, v in val.items()}})
        pd.DataFrame(history).to_csv(run / "history.csv", index=False)
        print(f"Epoch {epoch}: train={train['total_loss']:.6f}, inner-val={val['total_loss']:.6f}", flush=True)
        if val["total_loss"] < best - config["training"]["min_delta"]:
            best, stale = val["total_loss"], 0
            atomic_save({"kind": "autoencoder_pretrain_v1", "projector": args.projector,
                         "config": config, **context, "feature_signature": feature_signature(config),
                         "epoch": epoch, "patch_state": {k: v.detach().cpu() for k, v in model.affinity.patches.state_dict().items()},
                         "autoencoder_state": {k: v.detach().cpu() for k, v in autoencoder.state_dict().items()},
                         "inner_train_proteins": [k for k in proteins if k not in held_out],
                         "inner_val_proteins": sorted(held_out),
                         "source_patch_checkpoint": str(args.patch_checkpoint) if args.patch_checkpoint else None,
                         "debug_protein_limit": args.limit_proteins}, run / "best.pt")
        else:
            stale += 1
        if stale >= config["training"]["patience"]:
            break
    checkpoint = torch.load(run / "best.pt", weights_only=True, map_location="cpu")
    autoencoder.load_state_dict(checkpoint["autoencoder_state"])
    final = fit_epoch(autoencoder, matrices["val"], s["pretrain_batch_size"], None,
                      generator, s["activity_threshold"], device)
    save_json({"dataset": args.dataset, "split": args.split, "seed": config["seed"],
               "best_epoch": checkpoint["epoch"], "pretraining_seconds": perf_counter() - started,
               "train_patches": len(matrices["train"]), "val_patches": len(matrices["val"]),
               "patch_encoder_source": str(args.patch_checkpoint) if args.patch_checkpoint else "shared seeded initialization (not affinity-trained)",
               **final}, run / "metrics.json")
    (run / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Saved {run / 'best.pt'}; extracted from affinity-train rows only (random splits can share protein identities)")


if __name__ == "__main__":
    main()
