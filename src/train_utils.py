"""AE-aware training, diagnostics and safe transfer of pretrained patch representations."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .metrics import regression_metrics
from .sae_losses import affinity_objective
from .utils import digest


def projector_label(config):
    kind = config["model"].get("projector", "mlp")
    return f"sae_{config['sae']['sparsity_mode']}" if kind == "sae" else kind


def feature_signature(config):
    protein = config["encoders"]["protein"]
    return digest({"protein": {k: protein[k] for k in ("model", "revision", "chunk_size")},
                   "patch_dim": config["model"]["patch_dim"], "patches": config["patches"]})


def experiment_signature(config):
    """Guard seed aggregation against different architectures, losses or training budgets."""
    kind = config["model"].get("projector", "mlp")
    ae = config.get("sae", {})
    keys = ["latent_dim", "hidden_dim", "reconstruction_weight"]
    if kind == "sae":
        keys += ["sparsity_mode", "topk", "sparsity_weight", "lambda_sparse"]
    training = {k: v for k, v in config["training"].items() if k not in ("output", "save_latents")}
    return digest({"features": feature_signature(config), "model": config["model"],
                   "sae": {k: ae.get(k) for k in keys} if kind != "mlp" else None,
                   "training": training, "fractions": config["data"]["fractions"],
                   "ligand": {k: v for k, v in config["encoders"]["ligand"].items() if k != "cache"}})


def default_pretrain_path(config, dataset, split, seed, projector=None):
    label = projector_label(config) if projector is None else (
        "dense_ae" if projector == "dense_ae" else f"sae_{config['sae']['sparsity_mode']}")
    return Path(config["training"]["output"]) / "pretrain" / dataset / label / split / f"seed_{seed}" / "best.pt"


def verify_transfer(checkpoint, config, context):
    for key in ("dataset", "split", "seed", "fingerprint"):
        if checkpoint.get(key) != context[key]:
            raise ValueError(f"Pretrained checkpoint differs in {key}; pretrain on this exact training split")
    saved = checkpoint.get("training_rows", checkpoint.get("splits", {}).get("train"))
    if saved is None or set(saved) != set(context["training_rows"]):
        raise ValueError("Pretrained training rows differ; refuse possible train/test leakage")
    signature = checkpoint.get("feature_signature", feature_signature(checkpoint["config"]))
    if signature != feature_signature(config):
        raise ValueError("Pretrained ESM/patch representation differs from current configuration")


def patch_state(checkpoint):
    if checkpoint.get("kind") == "autoencoder_pretrain_v1":
        return checkpoint["patch_state"]
    state = {k.removeprefix("affinity.patches."): v for k, v in checkpoint["state"].items()
             if k.startswith("affinity.patches.")}
    if not state:
        raise ValueError("Checkpoint contains no patch encoder")
    return state


def load_patch_reference(model, path, config, context):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    verify_transfer(checkpoint, config, context)
    model.affinity.patches.load_state_dict(patch_state(checkpoint), strict=True)
    return checkpoint


def apply_pretraining(model, config, context, sae_path=None, patch_path=None):
    if sae_path and patch_path and Path(sae_path).resolve() != Path(patch_path).resolve():
        raise ValueError("Use the same checkpoint for SAE and its patch representation")
    path = sae_path or patch_path
    if path is None:
        if config.get("sae", {}).get("freeze_after_pretrain") and model.affinity.projector_kind != "mlp":
            raise ValueError("Cannot freeze an untrained autoencoder: supply a pretraining checkpoint")
        return "end_to_end"
    checkpoint = load_patch_reference(model, path, config, context)
    if sae_path:
        if checkpoint.get("kind") != "autoencoder_pretrain_v1":
            raise ValueError("SAE transfer requires an autoencoder pretraining checkpoint")
        kind = config["model"]["projector"]
        if kind not in ("sae", "dense_ae") or checkpoint["projector"] != kind:
            raise ValueError("Pretrained autoencoder kind differs")
        keys = ("latent_dim", "hidden_dim") if kind == "dense_ae" else ("latent_dim", "hidden_dim", "sparsity_mode", "topk")
        if any(checkpoint["config"]["sae"][key] != config["sae"][key] for key in keys):
            raise ValueError("Pretrained latent architecture/sparsity differs")
        model.affinity.projection.autoencoder.load_state_dict(checkpoint["autoencoder_state"], strict=True)
        if config["sae"]["freeze_after_pretrain"]:
            model.affinity.projection.autoencoder.requires_grad_(False)
            config["sae"]["frozen"] = True
        config["sae"]["pretrained_checkpoint"] = str(Path(sae_path).resolve())
    if config["model"].get("freeze_patch_encoder", False):
        model.affinity.patches.requires_grad_(False)
    print(f"Loaded {'autoencoder + patches' if sae_path else 'patches'} from {path}")
    return "pretrained_frozen" if sae_path and config["sae"]["freeze_after_pretrain"] else "pretrained_finetune" if sae_path else "shared_patches"


class LatentStats:
    """Streaming moments on CPU; memory O(latent_dim) unless an export is requested."""
    def __init__(self, threshold=1e-6, topk=None, export=False):
        if threshold < 0:
            raise ValueError("Activity threshold must be nonnegative")
        self.threshold, self.topk, self.export = threshold, topk, export
        self.count = 0
        self.sum = self.square = None
        self.zeros = self.inactive = self.l1 = self.reconstruction = self.sparse = 0.
        self.buffers = {k: [] for k in ("z", "row_id", "patch_index", "attention", "protein_id", "ligand_id")}

    def update(self, aux, reconstruction, sparse, records=None):
        if "latent" not in aux:
            return
        z = aux["latent"].detach().to("cpu", torch.float64)
        if z.ndim != 2 or not len(z) or not torch.isfinite(z).all():
            raise ValueError("Invalid latent matrix")
        n = len(z)
        if self.sum is None:
            self.sum = torch.zeros(z.shape[1], dtype=torch.float64)
            self.square = self.sum.clone()
        self.sum += z.sum(0)
        self.square += z.square().sum(0)
        self.zeros += z.eq(0).sum().item()
        self.inactive += z.abs().le(self.threshold).sum().item()
        self.l1 += z.abs().sum().item()
        self.reconstruction += float(reconstruction.detach()) * n
        self.sparse += float(sparse.detach()) * n
        self.count += n
        if self.export:
            self.buffers["z"].append(z.float().numpy())
            if records is not None:
                counts = aux["patch_counts"].detach().cpu().tolist()
                if len(counts) != len(records) or sum(counts) != n:
                    raise ValueError("Latents and record metadata are misaligned")
                self.buffers["row_id"].append(np.repeat([r["row_id"] for r in records], counts))
                self.buffers["patch_index"].append(np.concatenate([np.arange(c) for c in counts]))
                self.buffers["attention"].append(np.concatenate([
                    aux["attention"][i, :c].detach().cpu().numpy() for i, c in enumerate(counts)]))
                for key in ("protein_id", "ligand_id"):
                    self.buffers[key].append(np.repeat([str(r[key]) for r in records], counts))

    def metrics(self):
        if not self.count:
            return {}
        dims = len(self.sum)
        variance = (self.square / self.count - (self.sum / self.count).square()).clamp_min(0)
        inactive_ratio = self.inactive / (self.count * dims)
        return {"reconstruction_mse": self.reconstruction / self.count,
                "sparsity_loss": self.sparse / self.count,
                "latent_patches": self.count, "latent_dim": dims,
                "latent_sparsity_ratio": inactive_ratio,
                "latent_fraction_zeros": self.zeros / (self.count * dims),
                "latent_active_ratio": 1 - inactive_ratio,
                "latent_mean_active": dims * (1 - inactive_ratio),
                "latent_mean_l1_norm": self.l1 / self.count,
                "latent_variance": variance.mean().item(),
                "nominal_topk_ratio": self.topk / dims if self.topk is not None else None}

    def save(self, path):
        if not self.count or not self.export:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.concatenate(v) for k, v in self.buffers.items() if v}
        np.savez_compressed(path, **arrays, activity_threshold=np.array(self.threshold))


def run_epoch(model, loader, device, mean, std, optimizer=None, gradient_clip=5., latent_path=None):
    model.train(optimizer is not None)
    predictions, labels, rows = [], [], []
    affinity_sum = total_sum = 0.
    config = model.config
    sae = config.get("sae", {})
    topk = sae.get("topk") if config["model"].get("projector") == "sae" and sae.get("sparsity_mode") == "topk" else None
    stats = LatentStats(sae.get("activity_threshold", 1e-6), topk, export=latent_path is not None)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = perf_counter()
    for records, target in loader:
        target = target.to(device)
        normalized = (target - mean) / std
        with torch.set_grad_enabled(optimizer is not None):
            output, aux = model(records, return_aux=True)
            loss, parts = affinity_objective(output, normalized, aux, config)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite total loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
                optimizer.step()
        affinity_sum += parts["affinity"].item() * len(records)
        total_sum += loss.item() * len(records)
        stats.update(aux, parts["reconstruction"], parts["sparsity"], records)
        predictions.extend((output.detach().cpu().numpy() * std + mean).tolist())
        labels.extend(target.detach().cpu().tolist())
        rows.extend(r["row_id"] for r in records)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - started
    metrics = regression_metrics(labels, predictions)
    metrics.update(normalized_mse=affinity_sum / len(labels), total_loss=total_sum / len(labels),
                   elapsed_seconds=elapsed, **stats.metrics())
    if latent_path:
        stats.save(latent_path)
    return metrics, {"row_id": rows, "affinity": labels, "prediction": predictions}
