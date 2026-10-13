from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..metrics import regression_metrics
from ..sae_losses import affinity_objective
from ..train_utils import LatentStats
from ..utils import save_json


def run_epoch(model, loader, device, mean, std, optimizer=None, export=None):
    model.train(optimizer is not None)
    config = model.config
    topk = config["sae"]["topk"] if config["model"]["projector"] == "sae" and config["sae"]["sparsity_mode"] == "topk" else None
    stats = LatentStats(config["sae"]["activity_threshold"], topk)
    predictions, labels, identities = [], [], []
    affinity_sum = total_sum = 0.
    arrays = {key: [] for key in ("z", "complex_id", "trajectory_id", "frame_index", "patch_index", "anchor_atom_id", "spatial_attention")}
    temporal = []
    if device.type == "cuda": torch.cuda.synchronize(device)
    started = perf_counter()
    for records, target in loader:
        target = target.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            prediction, aux = model(records, return_aux=True)
            loss, terms = affinity_objective(prediction, (target - mean) / std, aux, config)
            if not torch.isfinite(loss): raise FloatingPointError("Nonfinite MD objective")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
        affinity_sum += terms["affinity"].item() * len(records)
        total_sum += loss.item() * len(records)
        stats.update(aux, terms["reconstruction"], terms["sparsity"])
        predictions.extend((prediction.detach().cpu().numpy() * std + mean).tolist())
        labels.extend(target.detach().cpu().tolist())
        identities.extend(r["complex_id"] for r in records)
        if export:
            if "latent" in aux:
                arrays["z"].append(aux["latent"].detach().cpu().numpy())
                for meta in aux["metadata"]:
                    t, k = meta["t"], meta["k"]
                    arrays["complex_id"].append(np.repeat(records[meta["complex_index"]]["complex_id"], t * k))
                    arrays["trajectory_id"].append(np.repeat(meta["trajectory_id"], t * k))
                    arrays["frame_index"].append(np.repeat(meta["frame_indices"].detach().cpu().numpy(), k))
                    arrays["patch_index"].append(np.tile(np.arange(k), t))
                    arrays["anchor_atom_id"].append(np.tile(meta["atom_ids"].detach().cpu().numpy(), t))
                    arrays["spatial_attention"].append(meta["attention"].detach().cpu().numpy().reshape(-1))
            for item in aux["temporal_attention"]:
                temporal.append({k: v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v for k, v in item.items()})
    if device.type == "cuda": torch.cuda.synchronize(device)
    metrics = regression_metrics(labels, predictions)
    metrics.update(normalized_mse=affinity_sum / len(labels), total_loss=total_sum / len(labels),
                   elapsed_seconds=perf_counter() - started, **stats.metrics())
    if export:
        path = Path(export); path.parent.mkdir(parents=True, exist_ok=True)
        if arrays["z"]:
            np.savez_compressed(path, **{k: np.concatenate(v) for k, v in arrays.items()},
                                activity_threshold=np.array(config["sae"]["activity_threshold"]))
        save_json(temporal, path.with_suffix(".temporal.json"))
    return metrics, {"complex_id": identities, "affinity": labels, "prediction": predictions}


def experiment_name(config):
    kind = config["model"]["projector"]
    if kind == "sae": kind += "_" + config["sae"]["sparsity_mode"]
    return kind + "_" + config["model"]["scorer"] + "_" + config["dynamic"]["temporal_mode"]
