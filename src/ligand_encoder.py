from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

from .utils import atomic_save, digest


class LigandEncoder(nn.Module):
    """ChemBERTa global mean over non-special, non-padding SMILES tokens."""
    def __init__(self, model="DeepChem/ChemBERTa-77M-MTR", revision="main",
                 freeze=True, cache=None, max_length=512):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
        self.backbone = AutoModel.from_pretrained(model, revision=revision, add_pooling_layer=False)
        self.dim = self.backbone.config.hidden_size
        self.frozen = freeze
        self.cache = Path(cache) if cache and freeze else None
        self.max_length = max_length
        capacity = self.backbone.config.max_position_embeddings - self.backbone.config.pad_token_id - 1
        if not 3 <= max_length <= capacity:
            raise ValueError(f"max_length must be in [3,{capacity}]")
        self.identity = {"model": model, "revision": revision,
                         "commit": getattr(self.backbone.config, "_commit_hash", None),
                         "max_length": max_length, "pool": "non-special-mean-v1"}
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

    def forward(self, smiles):
        device = next(self.backbone.parameters()).device
        path = self.cache / (digest([self.identity, smiles]) + ".pt") if self.cache else None
        if path and path.exists():
            result = torch.load(path, map_location=device, weights_only=True)
        else:
            tokens = self.tokenizer(smiles, return_tensors="pt", return_special_tokens_mask=True,
                                    truncation=False)
            if tokens["input_ids"].shape[1] > self.max_length:
                raise ValueError("SMILES exceeds max_length; do not silently truncate a molecule")
            special = tokens.pop("special_tokens_mask").to(device).bool()
            tokens = {k: v.to(device) for k, v in tokens.items()}
            mask = tokens["attention_mask"].bool() & ~special
            if not mask.any():
                raise ValueError("SMILES contains no content tokens")
            with torch.no_grad() if self.frozen else nullcontext():
                result = self.backbone(**tokens).last_hidden_state[mask].mean(0)
            if path:
                atomic_save(result.detach().cpu(), path)
        assert result.shape == (self.dim,)
        return result
