from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from .patch_selection import farthest_point_sampling, select_patches

STRUCTURE_DIM = 10


@lru_cache(maxsize=256)
def load_graph(path):
    if not path:
        return None
    if not Path(path).exists():
        warnings.warn(f"Graph missing: {path}; using sequence patches", stacklevel=2)
        return None
    # Only tensor dictionaries are accepted; never unpickle arbitrary objects.
    if Path(path).suffix == ".npz":
        import numpy as np
        with np.load(path, allow_pickle=False) as obj:
            graph = {k: torch.as_tensor(obj[k]) for k in obj.files}
    else:
        graph = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(graph, dict) or not all(isinstance(v, torch.Tensor) for v in graph.values()):
        raise ValueError("Graph must be a tensor-only dictionary (.pt or .npz)")
    return graph


def _moments(values, device):
    if values is None or values.numel() == 0:
        return torch.zeros(2, device=device)
    values = values.float().reshape(-1)
    if not torch.isfinite(values).all():
        raise ValueError("Non-finite graph features")
    return torch.stack((values.mean(), values.std(unbiased=False)))


def structural_summary(graph, members, n, device):
    def nodes(key, vectors=False):
        value = graph.get(key)
        if value is None:
            return None
        if value.shape[0] != n:
            raise ValueError(f"{key} must have {n} nodes")
        value = value[members]
        if vectors:
            if value.shape[-1] != 3:
                raise ValueError(f"{key} must end with xyz dimension")
            value = value.norm(dim=-1)
        return value

    ns, nv = nodes("node_s"), nodes("node_v", True)
    edge_index = graph.get("edge_index")
    es = ev = None
    degree = torch.zeros(len(members), device=device)
    if edge_index is not None:
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must be [2,E]")
        edge_index = edge_index.long()
        if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= n):
            raise ValueError("edge_index out of bounds")
        incident = torch.isin(edge_index[0], members) & torch.isin(edge_index[1], members)
        for key in ("edge_s", "edge_v"):
            if key in graph and len(graph[key]) != edge_index.shape[1]:
                raise ValueError(f"{key} is not aligned with edge_index")
        if "edge_s" in graph:
            es = graph["edge_s"][incident]
        if "edge_v" in graph:
            if graph["edge_v"].shape[-1] != 3:
                raise ValueError("edge_v must end with xyz dimension")
            ev = graph["edge_v"][incident].norm(dim=-1)
        degree = torch.bincount(edge_index[0, incident], minlength=n).float()[members]
    return torch.cat([_moments(v, device) for v in (ns, nv, es, ev, degree)])


class PatchEncoder(nn.Module):
    """Patch construction followed by a shared learned fixed-dimensional encoder."""
    def __init__(self, protein_dim, patch_dim=256, window=32, radius=8., max_patches=32, k=8):
        super().__init__()
        if min(window, max_patches, k) < 1 or radius <= 0 or k > max_patches:
            raise ValueError("Require window>0, radius>0, 1<=K<=max_patches")
        self.window, self.radius, self.max_patches, self.k = window, radius, max_patches, k
        self.encoder = nn.Sequential(nn.LayerNorm(protein_dim + STRUCTURE_DIM),
                                     nn.Linear(protein_dim + STRUCTURE_DIM, patch_dim), nn.GELU())
        self._warned = set()

    def _warn(self, kind):
        if kind not in self._warned:
            warnings.warn(kind, stacklevel=3)
            self._warned.add(kind)

    def forward(self, residues, graph=None):
        assert residues.ndim == 2 and len(residues) > 0
        device = residues.device
        raw, centers = [], None
        if graph:
            graph = {k: v.to(device) for k, v in graph.items()}
            coords = graph.get("coords", graph.get("pos"))
            if coords is not None and (coords.ndim != 2 or coords.shape[1] != 3
                                       or not torch.isfinite(coords).all()):
                raise ValueError("coords/pos must be finite [N,3], in Angstrom")
            n = len(coords) if coords is not None else next(
                (len(graph[key]) for key in ("node_s", "node_v", "residue_index") if key in graph), 0)
            if not n:
                self._warn("Graph has no usable nodes; using sequence windows")
                graph = None
            else:
                if coords is not None:
                    anchors = farthest_point_sampling(coords.float(), self.max_patches)
                    centers = coords[anchors].float()
                    neighborhoods = [torch.where((coords - coords[i]).norm(dim=-1) <= self.radius)[0]
                                     for i in anchors]
                else:
                    self._warn("No 3D coordinates: using contiguous graph-node groups (not 3D patches)")
                    neighborhoods = list(torch.tensor_split(torch.arange(n, device=device),
                                                           min(n, self.max_patches)))
                alignment = graph.get("residue_index")
                if alignment is not None:
                    if alignment.shape != (n,) or alignment.dtype not in (torch.int32, torch.int64):
                        raise ValueError("residue_index must be integer [N], zero-based")
                    if alignment.min() < 0 or alignment.max() >= len(residues):
                        raise ValueError("Graph/sequence alignment out of bounds")
                else:
                    self._warn("No residue_index: graph patches use global ESM mean; no alignment inferred")
                for members in neighborhoods:
                    sequence = residues[alignment[members]].mean(0) if alignment is not None else residues.mean(0)
                    raw.append(torch.cat((sequence, structural_summary(graph, members, n, device))))
        if not graph:
            self._warn("No usable structure: sequence-window fallback (not a structural reproduction)")
            for part in residues.split(self.window):
                raw.append(torch.cat((part.mean(0), torch.zeros(STRUCTURE_DIM, device=device))))
            if len(raw) > self.max_patches:
                indices = torch.linspace(0, len(raw) - 1, self.max_patches).round().long().tolist()
                raw = [raw[i] for i in indices]
        raw = torch.stack(raw)
        selected, _ = select_patches(raw, self.k, centers)
        patches = self.encoder(selected)
        assert patches.ndim == 2 and 1 <= len(patches) <= self.k
        return patches
