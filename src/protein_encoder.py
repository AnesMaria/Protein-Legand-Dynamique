from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .utils import atomic_save, digest


class ProteinEncoder(nn.Module):
    """ESM-2 per-residue encoding; chunk long sequences without dropping residues."""
    def __init__(self, model, revision="main", freeze=True, cache=None, chunk_size=1000):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.backbone = AutoModel.from_pretrained(model, revision=revision, add_pooling_layer=False)
        self.dim = self.backbone.config.hidden_size
        self.frozen = freeze
        self.cache = Path(cache) if cache and freeze else None
        self.chunk_size = chunk_size
        capacity = self.backbone.config.max_position_embeddings - 2
        if not 1 <= chunk_size <= capacity:
            raise ValueError(f"chunk_size must be in [1,{capacity}]")
        self.identity = {"model": model, "revision": revision,
                         "commit": getattr(self.backbone.config, "_commit_hash", None),
                         "chunk": chunk_size, "algorithm": "residue-v1"}
        if self.identity["commit"]:
            self.identity["revision"] = self.identity["commit"]
        if freeze:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, sequence, pooling="residue"):
        if pooling not in ("residue", "mean"):
            raise ValueError("pooling must be residue or mean")
        sequence = "".join(sequence.split()).upper()
        if not sequence:
            raise ValueError("Empty protein sequence")
        device = next(self.backbone.parameters()).device
        path = self.cache / (digest([self.identity, sequence]) + ".pt") if self.cache else None
        if path and path.exists():
            result = torch.load(path, map_location=device, weights_only=True)
        else:
            chunks = []
            with torch.no_grad() if self.frozen else nullcontext():
                for offset in range(0, len(sequence), self.chunk_size):
                    part = sequence[offset:offset + self.chunk_size]
                    tokens = self.tokenizer(part, return_tensors="pt", return_special_tokens_mask=True)
                    special = tokens.pop("special_tokens_mask").to(device).bool()
                    tokens = {k: v.to(device) for k, v in tokens.items()}
                    hidden = self.backbone(**tokens).last_hidden_state
                    mask = tokens["attention_mask"].bool() & ~special
                    chunk = hidden[mask]
                    if len(chunk) != len(part):
                        raise ValueError("ESM token/residue alignment failed")
                    chunks.append(chunk)
                result = torch.cat(chunks)
            if path:
                atomic_save(result.detach().cpu(), path)
        assert result.shape == (len(sequence), self.dim)
        return result.mean(0) if pooling == "mean" else result
