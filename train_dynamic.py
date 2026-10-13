import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
import yaml

from src.dynamic.data import read_manifest, split_complexes, fingerprint, loader
from src.dynamic.model import DynamicAffinityModel
from src.dynamic.engine import run_epoch, experiment_name
from src.utils import load_config, seed_everything, choose_device, atomic_save, save_json, digest


def main():
    p = argparse.ArgumentParser(description="Train on full MD complexes, never split frames/replicas across folds")
    p.add_argument("--config", default="dynamic_config.yaml")
    p.add_argument("--manifest", type=Path)
    p.add_argument("--model", choices=["classical", "quantum"])
    p.add_argument("--projector", choices=["mlp", "sae", "dense_ae"])
    p.add_argument("--sparsity-mode", choices=["l1", "topk"])
    p.add_argument("--temporal", choices=["gru", "mean", "first"])
    p.add_argument("--split", choices=["random", "cold_protein", "cold_ligand", "cold_both"], default="random")
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--output", type=Path)
    p.add_argument("--sae-checkpoint", type=Path)
    p.add_argument("--graph-checkpoint", type=Path)
    p.add_argument("--freeze-sae", action="store_true")
    p.add_argument("--freeze-graph", action="store_true", help="Use a fixed common spatial representation in controlled ablations")
    args = p.parse_args()
    config = copy.deepcopy(load_config(args.config))
    if args.manifest: config["data"]["manifest"] = str(args.manifest)
    if args.model: config["model"]["scorer"] = args.model
    if args.projector: config["model"]["projector"] = args.projector
    if args.sparsity_mode: config["sae"]["sparsity_mode"] = args.sparsity_mode
    if args.temporal: config["dynamic"]["temporal_mode"] = args.temporal
    if args.seed is not None: config["seed"] = args.seed
    if args.epochs is not None: config["training"]["epochs"] = args.epochs
    if config["training"]["epochs"] < 1: p.error("epochs must be positive")
    if args.freeze_sae and not args.sae_checkpoint: p.error("--freeze-sae requires trained weights")
    if args.sae_checkpoint and args.graph_checkpoint: p.error("A SAE checkpoint already includes its spatial encoder")
    seed_everything(config["seed"])
    device = choose_device(config["device"])
    frame = read_manifest(config["data"]["manifest"])
    splits = split_complexes(frame, args.split, config["seed"], config["data"]["fractions"])
    data_hash = fingerprint(frame)
    output = args.output or Path(config["training"]["output"]) / experiment_name(config) / args.split / f"seed_{config['seed']}"
    output.mkdir(parents=True, exist_ok=False)
    model = DynamicAffinityModel(config).to(device)
    if config["model"]["scorer"] == "quantum": model.scoring_features.circuit.to("cpu")
    if config["dynamic"]["language_context"]:
        for name in ("protein", "ligand"):
            commit = getattr(model, name + "_encoder").identity.get("commit")
            if commit: config["encoders"][name]["revision"] = commit
    if args.sae_checkpoint:
        from src.dynamic.pretrain import restore_pretraining
        restore_pretraining(model, args.sae_checkpoint, config, data_hash, args.split,
                            splits["train"].complex_id.tolist(), freeze=args.freeze_sae)
    elif args.graph_checkpoint:
        from src.dynamic.pretrain import restore_pretraining
        restore_pretraining(model, args.graph_checkpoint, config, data_hash, args.split,
                            splits["train"].complex_id.tolist(), graph_only=True)
    if args.freeze_graph: model.patches.requires_grad_(False)
    config["runtime"] = {"freeze_graph": args.freeze_graph, "freeze_sae": args.freeze_sae,
                          "sae_checkpoint": str(args.sae_checkpoint) if args.sae_checkpoint else None,
                          "graph_checkpoint": str(args.graph_checkpoint) if args.graph_checkpoint else None}
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config["training"]["learning_rate"], weight_decay=config["training"]["weight_decay"], foreach=False)
    mean = float(splits["train"].affinity.mean()); std = float(splits["train"].affinity.std(ddof=0))
    if std < 1e-8:
        import warnings
        warnings.warn("Train labels are constant: use unit target scale; this is not a meaningful predictive benchmark")
        std = 1.
    loaders = {k: loader(v, config, k == "train") for k, v in splits.items()}
    split_ids = {k: v.complex_id.tolist() for k, v in splits.items()}
    save_json(split_ids, output / "splits.json")
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    best, stale, history = float("inf"), 0, []
    started = perf_counter()
    for epoch in range(1, config["training"]["epochs"] + 1):
        train, _ = run_epoch(model, loaders["train"], device, mean, std, optimizer)
        val, _ = run_epoch(model, loaders["val"], device, mean, std)
        history.append({"epoch": epoch, **{"train_" + k: v for k, v in train.items()},
                        **{"val_" + k: v for k, v in val.items()}})
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        print(f"Epoch {epoch}: train MSE={train['normalized_mse']:.5f}, val MSE={val['normalized_mse']:.5f}", flush=True)
        if val["normalized_mse"] < best - config["training"]["min_delta"]:
            best, stale = val["normalized_mse"], 0
            atomic_save({"kind": "dynamic_affinity_v1", "state": model.checkpoint_state(), "config": config,
                         "fingerprint": data_hash, "split": args.split, "splits": split_ids,
                         "manifest": str(Path(config["data"]["manifest"]).resolve()),
                         "mean": mean, "std": std, "epoch": epoch}, output / "best.pt")
        else: stale += 1
        if stale >= config["training"]["patience"]: break
    training_seconds = perf_counter() - started
    checkpoint = torch.load(output / "best.pt", weights_only=True, map_location="cpu")
    model.restore(checkpoint["state"])
    metrics, predictions = run_epoch(model, loaders["test"], device, mean, std, export=output / "latents_test.npz")
    signature_config = copy.deepcopy(config)
    signature_config.pop("seed"); signature_config.pop("runtime", None)
    signature_config["training"].pop("output")
    # Different seeds may use different pretrained checkpoints but must share the same protocol.
    metrics.update(dataset="+".join(sorted(set(frame.source))), target_type=frame.target_type.iloc[0], seed=config["seed"],
                   model=config["model"]["scorer"], projector=config["model"]["projector"],
                   sparsity_mode=config["sae"]["sparsity_mode"] if config["model"]["projector"] == "sae" else "none",
                   split=args.split, temporal_mode=config["dynamic"]["temporal_mode"], experiment=experiment_name(config),
                   training_protocol="pretrained" if args.sae_checkpoint else "shared_graph" if args.graph_checkpoint else "end_to_end",
                   patch_encoder_frozen=args.freeze_graph, autoencoder_frozen=args.freeze_sae,
                   trainable_parameters=parameters, training_seconds=training_seconds,
                   inference_seconds=metrics["elapsed_seconds"], experiment_signature=digest(signature_config),
                   dataset_fingerprint=data_hash, best_epoch=checkpoint["epoch"],
                   n_train=len(splits["train"]), n_val=len(splits["val"]), n_test=len(splits["test"]))
    save_json(metrics, output / "metrics.json")
    pd.DataFrame(predictions).to_csv(output / "predictions.csv", index=False)
    print(f"Completed: {output}; experimental endpoint {metrics['target_type']}; test complexes={metrics['n']}")


if __name__ == "__main__": main()
