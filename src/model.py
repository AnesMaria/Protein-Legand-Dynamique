from __future__ import annotations

import torch
from torch import nn

from .patch_encoder import PatchEncoder, load_graph
from .utils import show_shapes


class PatchProjection(nn.Module):
    """Original baseline MLP, retained without numerical changes."""
    def __init__(self, patch_dim, hidden, n_qubits, dropout):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(patch_dim, hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, n_qubits))

    def forward(self, patches):
        return self.network(patches)


class AffinityModel(nn.Module):
    def __init__(self, protein_dim, ligand_dim, config, variant="classical"):
        super().__init__()
        if variant not in ("classical", "quantum"):
            raise ValueError("variant must be classical or quantum")
        c = config["model"]
        self.patches = PatchEncoder(protein_dim, c["patch_dim"], **config["patches"])
        self.projection = PatchProjection(c["patch_dim"], c["projection_hidden"],
                                          c["n_qubits"], c["dropout"])
        self.projector_kind = c.get("projector", "mlp")
        if self.projector_kind not in ("mlp", "sae", "dense_ae"):
            raise ValueError("projector must be mlp, sae or dense_ae")
        if self.projector_kind != "mlp":
            from .sparse_autoencoder import AutoencoderProjection
            # Consume exactly the original MLP initialization above, then isolate AE RNG.
            # With the same seed, the patch encoder/scorer/head start identically in all ablations.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(config.get("seed", 0) + 104729)
                self.projection = AutoencoderProjection(c["patch_dim"], c["n_qubits"], config["sae"],
                                                       dense=self.projector_kind == "dense_ae")
            if config["sae"].get("frozen", False):
                self.projection.autoencoder.requires_grad_(False)
        if c.get("freeze_patch_encoder", False):
            self.patches.requires_grad_(False)
        if variant == "quantum":
            from .quantum_layer import QuantumLayer
            self.scoring_features = QuantumLayer(c["n_qubits"], c["quantum_layers"], c["angle_encoding"])
        else:
            self.scoring_features = nn.Sequential(nn.Linear(c["n_qubits"], c["n_qubits"]), nn.Tanh())
        self.score = nn.Linear(c["n_qubits"], 1)
        dims = [c["patch_dim"] + ligand_dim, *c["head_hidden"], 1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.extend([nn.GELU(), nn.Dropout(c["dropout"])])
        self.head = nn.Sequential(*layers)
        self._shown = False

    def encode_patches(self, residues, graphs=None):
        graphs = graphs if graphs is not None else [None] * len(residues)
        if len(residues) != len(graphs):
            raise ValueError("Batch lengths disagree")
        return [self.patches(r, g) for r, g in zip(residues, graphs)]

    def forward(self, residues, ligands, graphs=None, return_attention=False, return_aux=False):
        graphs = graphs if graphs is not None else [None] * len(residues)
        if len(residues) != len(ligands) or len(graphs) != len(residues):
            raise ValueError("Batch lengths disagree")
        patches = self.encode_patches(residues, graphs)
        padded = nn.utils.rnn.pad_sequence(patches, batch_first=True)
        mask = torch.arange(padded.shape[1], device=padded.device)[None] < torch.tensor(
            [len(p) for p in patches], device=padded.device)[:, None]
        aux = {}
        if self.projector_kind == "mlp":
            projected = self.projection(padded)  # Original padding/dropout behavior preserved.
        else:
            valid_projection, aux = self.projection.forward_with_aux(padded[mask])
            projected = padded.new_zeros((*mask.shape, valid_projection.shape[-1]))
            projected[mask] = valid_projection
            aux["patch_counts"] = torch.tensor([len(p) for p in patches], device=padded.device)
        # Score actual patches only; padded patches never consume quantum evaluations.
        scores = torch.full(mask.shape, -torch.inf, device=padded.device, dtype=padded.dtype)
        scoring_features = self.scoring_features(projected[mask])
        scores[mask] = self.score(scoring_features).squeeze(-1)
        attention = scores.softmax(dim=1)
        protein = (attention.unsqueeze(-1) * padded).sum(dim=1)
        fusion = torch.cat([protein, ligands], dim=-1)
        prediction = self.head(fusion).squeeze(-1)
        assert prediction.shape == (len(residues),) and torch.isfinite(prediction).all()
        if not self._shown:
            show_shapes(residues=residues[0], ligand=ligands, patches=padded,
                        projection=projected, scoring_features=scoring_features,
                        scores=scores, attention=attention,
                        protein=protein, fusion=fusion, prediction=prediction)
            if aux:
                show_shapes(latent=aux["latent"], reconstructed=aux["reconstructed"])
            self._shown = True
        if return_aux:
            aux["attention"] = attention
            return prediction, aux
        return (prediction, attention) if return_attention else prediction


class EncodedAffinityModel(nn.Module):
    """Online encoding or disk cache; supports optionally unfrozen backbones."""
    def __init__(self, config, variant="classical"):
        super().__init__()
        from .protein_encoder import ProteinEncoder
        from .ligand_encoder import LigandEncoder
        self.protein_encoder = ProteinEncoder(**config["encoders"]["protein"])
        self.ligand_encoder = LigandEncoder(**config["encoders"]["ligand"])
        self.affinity = AffinityModel(self.protein_encoder.dim, self.ligand_encoder.dim, config, variant)
        self.config = config

    def encode_patch_records(self, records):
        residues = [self.protein_encoder(r["sequence"]) for r in records]
        graphs = [load_graph(r.get("graph_path", "")) for r in records]
        return self.affinity.encode_patches(residues, graphs)

    def forward(self, records, return_aux=False):
        residues = [self.protein_encoder(r["sequence"]) for r in records]
        ligands = torch.stack([self.ligand_encoder(r["smiles"]) for r in records])
        graphs = [load_graph(r.get("graph_path", "")) for r in records]
        return self.affinity(residues, ligands, graphs, return_aux=return_aux)

    def checkpoint_state(self):
        # Frozen public backbones are reconstructed by their exact commit, not duplicated in checkpoints.
        state = self.state_dict()
        for name in ("protein_encoder", "ligand_encoder"):
            if getattr(self, name).frozen:
                state = {k: v for k, v in state.items() if not k.startswith(name + ".backbone.")}
        return {k: v.detach().cpu() for k, v in state.items()}

    def restore(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        allowed = tuple(name + ".backbone." for name in ("protein_encoder", "ligand_encoder")
                        if getattr(self, name).frozen)
        if unexpected or any(not k.startswith(allowed) for k in missing):
            raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
