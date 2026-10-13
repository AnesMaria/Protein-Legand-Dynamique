"""Shared training/evaluation helpers; labels are standardized on train only."""
from __future__ import annotations

import numpy as np
import torch

from .metrics import regression_metrics


def run_epoch(model, loader, device, mean, std, optimizer=None, gradient_clip=5.):
    model.train(optimizer is not None)
    predictions, labels, rows = [], [], []
    loss_sum = 0.
    for records, target in loader:
        target = target.to(device)
        normalized = (target - mean) / std
        with torch.set_grad_enabled(optimizer is not None):
            output = model(records)
            loss = torch.nn.functional.mse_loss(output, normalized)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite MSE")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
                optimizer.step()
        loss_sum += loss.item() * len(records)
        predictions.extend((output.detach().cpu().numpy() * std + mean).tolist())
        labels.extend(target.detach().cpu().tolist())
        rows.extend(r["row_id"] for r in records)
    metrics = regression_metrics(labels, predictions)
    metrics["normalized_mse"] = loss_sum / len(labels)
    return metrics, {"row_id": rows, "affinity": labels, "prediction": predictions}


def pin_loaded_revisions(config, model):
    for name in ("protein", "ligand"):
        encoder = getattr(model, name + "_encoder")
        commit = encoder.identity.get("commit")
        if commit:
            config["encoders"][name]["revision"] = commit
            # Cache identity must depend on resolved commit, not alias spelling.
            encoder.identity["revision"] = commit


def table_fingerprint(frame):
    import hashlib
    from pathlib import Path
    from .utils import digest
    # Detect reordered or altered records when evaluating an existing split.
    graphs = {}
    for path in sorted(set(frame.graph_path) - {""}):
        obj = Path(path)
        graphs[path] = hashlib.sha256(obj.read_bytes()).hexdigest() if obj.exists() else "missing"
    records = frame[["protein_id", "ligand_id", "sequence", "smiles", "affinity", "graph_path"]].to_dict("records")
    return digest({"records": records, "graphs": graphs})
