"""Sparse compression of learned patch vectors; extraction/selection stay upstream."""
from __future__ import annotations

import torch
from torch import nn


class DenseAutoencoder(nn.Module):
    """Dense ablation: same architecture as SAE, without penalty or Top-K mask.

    ReLU is shared across ablations and may yield zeros even in this dense control.
    """
    def __init__(self, input_dim, latent_dim=64, hidden_dim=128):
        super().__init__()
        if min(input_dim, latent_dim, hidden_dim) < 1:
            raise ValueError("Autoencoder dimensions must be positive")
        self.input_dim, self.latent_dim = input_dim, latent_dim
        self.sparsity_mode = "none"
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, latent_dim), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, input_dim))

    def encode(self, patches):
        if patches.ndim != 2 or patches.shape[-1] != self.input_dim:
            raise ValueError(f"Expected valid patches [N,{self.input_dim}]")
        return self.encoder(patches)

    def forward(self, patches):
        latent = self.encode(patches)
        return latent, self.decoder(latent)


class SparseAutoencoder(DenseAutoencoder):
    def __init__(self, input_dim, latent_dim=64, hidden_dim=128, sparsity_mode="l1", topk=8):
        super().__init__(input_dim, latent_dim, hidden_dim)
        if sparsity_mode not in ("l1", "topk"):
            raise ValueError("sparsity_mode must be l1 or topk")
        if sparsity_mode == "topk" and not 1 <= topk <= latent_dim:
            raise ValueError("Top-K must satisfy 1 <= K <= latent_dim")
        self.sparsity_mode, self.topk = sparsity_mode, topk

    def encode(self, patches):
        latent = super().encode(patches)
        if self.sparsity_mode == "topk":
            # Stable sorting breaks ties deterministically; gradients flow through kept values.
            indices = torch.argsort(latent.abs(), dim=-1, descending=True, stable=True)[:, :self.topk]
            mask = torch.zeros_like(latent).scatter_(1, indices, 1.)
            latent = latent * mask
        return latent


class AutoencoderProjection(nn.Module):
    def __init__(self, input_dim, n_qubits, config, dense=False):
        super().__init__()
        kwargs = {"input_dim": input_dim, "latent_dim": config["latent_dim"],
                  "hidden_dim": config["hidden_dim"]}
        self.autoencoder = DenseAutoencoder(**kwargs) if dense else SparseAutoencoder(
            **kwargs, sparsity_mode=config["sparsity_mode"], topk=config["topk"])
        self.to_qubits = nn.Identity() if config["latent_dim"] == n_qubits else nn.Linear(config["latent_dim"], n_qubits)

    def forward_with_aux(self, patches):
        latent, reconstructed = self.autoencoder(patches)
        return self.to_qubits(latent), {"latent": latent, "reconstructed": reconstructed,
                                      "patch_target": patches.detach()}

    def forward(self, patches):
        return self.forward_with_aux(patches)[0]
