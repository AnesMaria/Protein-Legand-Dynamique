from pathlib import Path

import torch

from ..utils import digest


def spatial_signature(config):
    keys = ("max_frames", "frame_stride", "graph_dim", "graph_layers", "graph_radius", "patch_radius", "patches")
    return digest({"spatial": {k: config["dynamic"][k] for k in keys}, "patch_dim": config["model"]["patch_dim"]})


def restore_pretraining(model, path, config, fingerprint, split, training_complexes, freeze=False, graph_only=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") not in ("dynamic_sae_pretrain_v1", "dynamic_affinity_v1"):
        raise ValueError("Only dynamic checkpoints are compatible; static patch dimensions alone are insufficient")
    for key, expected in (("fingerprint", fingerprint), ("split", split)):
        if checkpoint[key] != expected: raise ValueError(f"Pretraining {key} differs; refuse split leakage")
    if checkpoint["config"]["seed"] != config["seed"]: raise ValueError("Pretraining seed differs")
    saved = checkpoint.get("training_complexes", checkpoint.get("splits", {}).get("train"))
    if set(saved or []) != set(training_complexes): raise ValueError("Different pretraining train complexes")
    if spatial_signature(checkpoint["config"]) != spatial_signature(config): raise ValueError("Different spatial/temporal preprocessing")
    if checkpoint["kind"] == "dynamic_sae_pretrain_v1":
        graph_state = checkpoint["graph_state"]
    else:
        graph_state = {k.removeprefix("patches."): v for k, v in checkpoint["state"].items() if k.startswith("patches.")}
    model.patches.load_state_dict(graph_state, strict=True)
    if not graph_only:
        if checkpoint["kind"] != "dynamic_sae_pretrain_v1" or config["model"]["projector"] == "mlp":
            raise ValueError("Autoencoder transfer requires a dynamic AE pretraining checkpoint")
        if checkpoint["config"]["model"]["projector"] != config["model"]["projector"]:
            raise ValueError("Dense and sparse pretrained encoders differ")
        keys = ["latent_dim", "hidden_dim"]
        if config["model"]["projector"] == "sae": keys += ["sparsity_mode", "topk"]
        if any(checkpoint["config"]["sae"][k] != config["sae"][k] for k in keys):
            raise ValueError("Different latent architecture/sparsity")
        model.projection.autoencoder.load_state_dict(checkpoint["autoencoder_state"])
        if freeze: model.projection.autoencoder.requires_grad_(False)
    print(f"Loaded {'graph only' if graph_only else 'spatial encoder + AE'}: {Path(path)}")
