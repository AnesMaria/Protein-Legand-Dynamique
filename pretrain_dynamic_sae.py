import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from pretrain_sae import fit_epoch
from src.dynamic.data import read_manifest, split_complexes, fingerprint, ComplexDataset
from src.dynamic.model import DynamicAffinityModel
from src.dynamic.pretrain import spatial_signature, restore_pretraining
from src.utils import load_config, seed_everything, choose_device, atomic_save, save_json


def main():
    p = argparse.ArgumentParser(description="Pretrain AE on dynamic patches from affinity-train complexes only")
    p.add_argument("--config", default="dynamic_config.yaml")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--projector", choices=["sae", "dense_ae"], default="sae")
    p.add_argument("--sparsity-mode", choices=["l1", "topk"])
    p.add_argument("--split", choices=["random", "cold_protein", "cold_ligand", "cold_both"], default="random")
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--graph-checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    config = copy.deepcopy(load_config(args.config))
    config["model"]["projector"] = args.projector
    config["model"]["scorer"] = "classical"  # No unused quantum circuit is trained here.
    if args.manifest: config["data"]["manifest"] = str(args.manifest)
    if args.seed is not None: config["seed"] = args.seed
    if args.sparsity_mode: config["sae"]["sparsity_mode"] = args.sparsity_mode
    if args.epochs < 1 or config["sae"]["lambda_sparse"] < 0: p.error("Invalid epochs/lambda_sparse")
    seed_everything(config["seed"])
    device = choose_device(config["device"])
    frame = read_manifest(config["data"]["manifest"])
    splits = split_complexes(frame, args.split, config["seed"], config["data"]["fractions"])
    data_hash = fingerprint(frame)
    train = splits["train"]
    keys = train.apply(lambda r: r.protein_key + ":" + r.ligand_key, axis=1)
    groups = sorted(set(keys))
    if len(groups) < 2: raise ValueError("At least 2 train groups needed for disjoint inner AE validation")
    held_out = {np.random.default_rng(config["seed"]).permutation(groups)[0]}
    model = DynamicAffinityModel(config).to(device).eval()
    if args.graph_checkpoint:
        restore_pretraining(model, args.graph_checkpoint, config, data_hash, args.split, train.complex_id.tolist(), graph_only=True)
    model.patches.requires_grad_(False)
    matrices = {"train": [], "val": []}
    with torch.no_grad():
        for i, record in enumerate(ComplexDataset(train, config["dynamic"])):
            destination = "val" if keys.iloc[i] in held_out else "train"
            for trajectory in record["trajectories"]:
                (patches, _, _), _ = model.patch_vectors(trajectory)
                matrices[destination].append(patches.flatten(0, 1).detach().cpu())
    matrices = {k: torch.cat(v) for k, v in matrices.items()}
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_save(matrices, args.output / "patches.pt")
    ae = model.projection.autoencoder
    ae.lambda_sparse = config["sae"]["lambda_sparse"] if args.projector == "sae" else 0.
    optimizer = torch.optim.AdamW(ae.parameters(), lr=config["training"]["learning_rate"])
    generator = torch.Generator().manual_seed(config["seed"])
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        tr = fit_epoch(ae, matrices["train"], 256, optimizer, generator, config["sae"]["activity_threshold"], device)
        val = fit_epoch(ae, matrices["val"], 256, None, generator, config["sae"]["activity_threshold"], device)
        history.append({"epoch": epoch, **{"train_" + k: v for k, v in tr.items()}, **{"val_" + k: v for k, v in val.items()}})
        pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
        print(f"AE epoch {epoch}: train={tr['total_loss']:.5f} val={val['total_loss']:.5f}", flush=True)
        if val["total_loss"] < best - config["training"]["min_delta"]:
            best, stale = val["total_loss"], 0
            atomic_save({"kind": "dynamic_sae_pretrain_v1", "config": config, "fingerprint": data_hash,
                         "split": args.split, "training_complexes": train.complex_id.tolist(), "epoch": epoch,
                         "spatial_signature": spatial_signature(config),
                         "inner_train_complexes": train.loc[~keys.isin(held_out), "complex_id"].tolist(),
                         "inner_val_complexes": train.loc[keys.isin(held_out), "complex_id"].tolist(),
                         "graph_state": {k: v.detach().cpu() for k, v in model.patches.state_dict().items()},
                         "autoencoder_state": {k: v.detach().cpu() for k, v in ae.state_dict().items()}}, args.output / "best.pt")
        else: stale += 1
        if stale >= config["training"]["patience"]: break
    save_json({"train_patches": len(matrices["train"]), "val_patches": len(matrices["val"]),
               "source": "affinity-train complex groups only, spatial encoder frozen",
               "graph_source": str(args.graph_checkpoint) if args.graph_checkpoint else "seeded initialization, not affinity-trained"},
              args.output / "provenance.json")
    (args.output / "config.yaml").write_text(yaml.safe_dump(config))
    print(args.output / "best.pt")


if __name__ == "__main__": main()
