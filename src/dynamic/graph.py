"""Classical distance-based message passing and stable trajectory patch identities."""
import math

import torch
from torch import nn

from ..patch_selection import farthest_point_sampling


class InteractionGraphEncoder(nn.Module):
    def __init__(self, hidden_dim=64, radius=5., layers=2, rbf_count=16):
        super().__init__()
        if min(hidden_dim, layers, rbf_count) < 1 or radius <= 0:
            raise ValueError("Invalid graph encoder dimensions/radius")
        self.radius = radius
        self.element = nn.Embedding(119, 16)
        self.component = nn.Embedding(2, 4)
        self.input = nn.Sequential(nn.Linear(21, hidden_dim), nn.SiLU())
        self.register_buffer("centers", torch.linspace(0., radius, rbf_count))
        self.messages = nn.ModuleList([nn.Sequential(nn.Linear(2 * hidden_dim + rbf_count + 1, hidden_dim),
                                                     nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(layers)])
        self.updates = nn.ModuleList([nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
                                                    nn.LayerNorm(hidden_dim)) for _ in range(layers)])

    def forward(self, positions, atomic_numbers, ligand_mask):
        distances = torch.cdist(positions, positions)
        cross = distances[~ligand_mask][:, ligand_mask]
        nearest = positions.new_zeros(len(positions))
        nearest[~ligand_mask] = cross.min(1).values
        nearest[ligand_mask] = cross.min(0).values
        # Per-node protein/ligand proximity also represents disappearance of contact edges.
        features = torch.cat([self.element(atomic_numbers), self.component(ligand_mask.long()),
                              (nearest / self.radius).clamp(max=10).unsqueeze(-1)], dim=-1)
        h = self.input(features)
        # Radius graph, not a chemical bond graph. Both edge directions are represented.
        edge = ((distances < self.radius) & ~torch.eye(len(positions), device=positions.device, dtype=torch.bool)).nonzero()
        source, destination = edge.T
        d = distances[source, destination]
        rbf = torch.exp(-((d[:, None] - self.centers) / (self.radius / len(self.centers))) ** 2)
        cross_flag = (ligand_mask[source] != ligand_mask[destination]).float().unsqueeze(-1)
        cutoff = .5 * (torch.cos(math.pi * d / self.radius) + 1)
        degree = torch.bincount(destination, minlength=len(positions)).clamp_min(1).to(h.dtype).unsqueeze(-1)
        for message, update in zip(self.messages, self.updates):
            value = message(torch.cat([h[source], h[destination], rbf, cross_flag], dim=-1)) * cutoff[:, None]
            aggregated = torch.zeros_like(h).index_add(0, destination, value) / degree
            h = h + update(torch.cat([h, aggregated], dim=-1))
        return h


class DynamicPatchEncoder(nn.Module):
    def __init__(self, graph_dim=64, patch_dim=256, graph_radius=5., graph_layers=2,
                 patch_radius=8., k=8):
        super().__init__()
        if patch_radius <= 0 or k < 1:
            raise ValueError("Invalid patch radius/K")
        self.graph = InteractionGraphEncoder(graph_dim, graph_radius, graph_layers)
        self.representation = nn.Sequential(nn.Linear(graph_dim, patch_dim), nn.GELU(), nn.LayerNorm(patch_dim))
        self.patch_radius, self.k = patch_radius, k

    def forward(self, trajectory):
        xyz, ligand = trajectory["positions"], trajectory["ligand_mask"]
        z = trajectory["atomic_numbers"]
        protein_atoms = (~ligand).nonzero().flatten()
        # FPS only once: anchors follow the same atom identities through all frames.
        reference = trajectory.get("reference_positions", xyz[0])
        anchors = protein_atoms[farthest_point_sampling(reference[protein_atoms], self.k)]
        patches, ligand_features = [], []
        for positions in xyz:
            h = self.graph(positions, z, ligand)
            distances = torch.cdist(positions[anchors], positions)
            raw = torch.stack([h[distances[i] <= self.patch_radius].mean(0) for i in range(len(anchors))])
            patches.append(self.representation(raw))
            ligand_features.append(h[ligand].mean(0))
        return torch.stack(patches), torch.stack(ligand_features), anchors
