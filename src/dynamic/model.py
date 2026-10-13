from __future__ import annotations

import torch
from torch import nn

from ..model import PatchProjection
from ..sparse_autoencoder import AutoencoderProjection
from ..utils import show_shapes
from .graph import DynamicPatchEncoder


class TemporalPooling(nn.Module):
    def __init__(self, input_dim, hidden_dim, mode="gru"):
        super().__init__()
        if mode not in ("gru", "mean", "first"):
            raise ValueError("Temporal mode must be gru, mean or first")
        self.mode = mode
        self.input = nn.Sequential(nn.Linear(input_dim + 2, hidden_dim), nn.GELU())
        if mode == "gru":
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
            self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, frames, times):
        if len(frames) != len(times) or len(frames) < 1:
            raise ValueError("Frame/time mismatch")
        if self.mode == "gru":
            elapsed = (times - times[0]) / (times[-1] - times[0]).clamp_min(1e-8)
            delta = torch.cat([times.new_zeros(1), torch.diff(times)])
            delta = delta / torch.diff(times).median().clamp_min(1e-8)
            timing = torch.stack([elapsed, delta], -1).to(frames.dtype)
        else:
            timing = frames.new_zeros(len(frames), 2)  # Mean control is permutation invariant.
        h = self.input(torch.cat([frames, timing], -1))
        if self.mode == "gru":
            h, _ = self.gru(h.unsqueeze(0))
            h = h.squeeze(0)
            weights = self.attention(h).squeeze(-1).softmax(0)
            return (weights[:, None] * h).sum(0), weights
        if self.mode == "first":
            weights = h.new_zeros(len(h)); weights[0] = 1
            return h[0], weights
        return h.mean(0), h.new_full((len(h),), 1 / len(h))


class DynamicAffinityModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c, d = config["model"], config["dynamic"]
        self.patches = DynamicPatchEncoder(d["graph_dim"], c["patch_dim"], d["graph_radius"],
                                           d["graph_layers"], d["patch_radius"], d["patches"])
        self.projector_kind = c["projector"]
        self.projection = PatchProjection(c["patch_dim"], c["projection_hidden"], c["n_qubits"], c["dropout"])
        if self.projector_kind not in ("mlp", "sae", "dense_ae"):
            raise ValueError("Unknown projector")
        if self.projector_kind != "mlp":
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(config["seed"] + 104729)
                self.projection = AutoencoderProjection(c["patch_dim"], c["n_qubits"], config["sae"],
                                                       dense=self.projector_kind == "dense_ae")
        if c["scorer"] == "quantum":
            from ..quantum_layer import QuantumLayer
            self.scoring_features = QuantumLayer(c["n_qubits"], c["quantum_layers"], c["angle_encoding"])
        elif c["scorer"] == "classical":
            self.scoring_features = nn.Sequential(nn.Linear(c["n_qubits"], c["n_qubits"]), nn.Tanh())
        else:
            raise ValueError("Unknown scorer")
        self.score = nn.Linear(c["n_qubits"], 1)
        self.temporal = TemporalPooling(c["patch_dim"] + d["graph_dim"], d["temporal_dim"], d["temporal_mode"])
        context_dim = 0
        if d["language_context"]:
            from ..protein_encoder import ProteinEncoder
            from ..ligand_encoder import LigandEncoder
            self.protein_encoder = ProteinEncoder(**config["encoders"]["protein"])
            self.ligand_encoder = LigandEncoder(**config["encoders"]["ligand"])
            context_dim = self.protein_encoder.dim + self.ligand_encoder.dim
        self.head = nn.Sequential(nn.Linear(d["temporal_dim"] + context_dim, c["head_hidden"][0]),
                                  nn.GELU(), nn.Dropout(c["dropout"]), nn.Linear(c["head_hidden"][0], 1))
        self._shown = False

    def patch_vectors(self, trajectory):
        device = next(self.patches.parameters()).device
        trajectory = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in trajectory.items()}
        return self.patches(trajectory), trajectory

    def forward(self, records, return_aux=False):
        representations, auxiliaries, metadata = [], [], []
        temporal_weights = []
        for complex_index, record in enumerate(records):
            replicas = []
            for replica_index, raw_trajectory in enumerate(record["trajectories"]):
                if self.temporal.mode == "first":
                    raw_trajectory = {**raw_trajectory, **{key: raw_trajectory[key][:1]
                                      for key in ("positions", "time", "frame_indices")}}
                (patches, ligand_features, anchors), trajectory = self.patch_vectors(raw_trajectory)
                t, k, dim = patches.shape
                flat = patches.reshape(t * k, dim)
                if self.projector_kind == "mlp":
                    projected, aux = self.projection(flat), {}
                else:
                    projected, aux = self.projection.forward_with_aux(flat)
                scores = self.score(self.scoring_features(projected)).reshape(t, k)
                weights = scores.softmax(-1)
                pooled = (weights[..., None] * patches).sum(1)
                frame_repr = torch.cat([pooled, ligand_features], -1)
                temporal_repr, time_weights = self.temporal(frame_repr, trajectory["time"])
                replicas.append(temporal_repr)
                temporal_weights.append({"complex_id": record["complex_id"],
                                         "trajectory_id": raw_trajectory["trajectory_id"],
                                         "frame_indices": trajectory["frame_indices"].detach().cpu(),
                                         "time": trajectory["time"].detach().cpu(),
                                         "time_unit": trajectory["time_unit"], "weights": time_weights})
                if aux:
                    auxiliaries.append(aux)
                    metadata.append({"complex_index": complex_index, "replica_index": replica_index,
                                     "trajectory_id": raw_trajectory["trajectory_id"], "t": t, "k": k,
                                     "frame_indices": trajectory["frame_indices"],
                                     "atom_ids": trajectory["atom_ids"][anchors], "attention": weights})
                if not self._shown:
                    show_shapes(coordinates=trajectory["positions"], dynamic_patches=patches,
                                projection=projected, spatial_attention=weights,
                                frame_features=frame_repr, temporal_representation=temporal_repr)
                    if aux:
                        show_shapes(latent=aux["latent"], reconstructed=aux["reconstructed"])
                    self._shown = True
            # Each complex contributes one label/loss, irrespective of its replica count.
            representation = torch.stack(replicas).mean(0)
            if self.config["dynamic"]["language_context"]:
                if not record.get("sequence") or not record.get("smiles"):
                    raise ValueError("ESM/ChemBERTa context needs true sequence and SMILES in the manifest")
                representation = torch.cat([representation, self.protein_encoder(record["sequence"], pooling="mean"),
                                             self.ligand_encoder(record["smiles"])])
            representations.append(representation)
        prediction = self.head(torch.stack(representations)).squeeze(-1)
        assert prediction.shape == (len(records),) and torch.isfinite(prediction).all()
        aux = {key: torch.cat([a[key] for a in auxiliaries]) for key in ("latent", "reconstructed", "patch_target")} if auxiliaries else {}
        aux.update(metadata=metadata, temporal_attention=temporal_weights)
        return (prediction, aux) if return_aux else prediction

    def checkpoint_state(self):
        state = self.state_dict()
        for name in ("protein_encoder", "ligand_encoder"):
            if hasattr(self, name) and getattr(self, name).frozen:
                state = {k: v for k, v in state.items() if not k.startswith(name + ".backbone.")}
        return {k: v.detach().cpu() for k, v in state.items()}

    def restore(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        allowed = tuple(name + ".backbone." for name in ("protein_encoder", "ligand_encoder")
                        if hasattr(self, name) and getattr(self, name).frozen)
        if unexpected or any(not key.startswith(allowed) for key in missing):
            raise ValueError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
