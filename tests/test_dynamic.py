import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.dynamic.data import read_manifest, read_trajectory, split_complexes, validate_trajectory
from src.dynamic.prepare import save_canonical, crop_trajectory
from src.dynamic.model import DynamicAffinityModel, TemporalPooling
from src.dynamic.graph import InteractionGraphEncoder
from src.sae_losses import affinity_objective
from src.utils import load_config, seed_everything
from test_encoders import small_backbones


def small_config(projector="sae", scorer="quantum", temporal="gru"):
    config = load_config("dynamic_config.yaml")
    config["dynamic"].update(graph_dim=12, graph_layers=1, patches=3, temporal_dim=12, temporal_mode=temporal)
    config["model"].update(projector=projector, scorer=scorer, patch_dim=16, dropout=0., head_hidden=[12])
    config["sae"].update(latent_dim=8, hidden_dim=12, topk=3)
    return config


def trajectory(t=4):
    xyz = torch.randn(t, 7, 3)
    return {"positions": xyz, "reference_positions": xyz[0].clone(), "atomic_numbers": torch.tensor([6, 7, 8, 6, 6, 8, 7]),
            "ligand_mask": torch.tensor([False] * 5 + [True] * 2), "time": torch.arange(t, dtype=torch.float64),
            "frame_indices": torch.arange(t), "atom_ids": torch.arange(7), "time_unit": "frame_index", "trajectory_id": "rep0"}


@pytest.mark.parametrize("projector,scorer", [("mlp", "classical"), ("mlp", "quantum"),
                                              ("sae", "classical"), ("sae", "quantum"), ("dense_ae", "quantum")])
def test_dynamic_gradient_roundtrip(projector, scorer):
    torch.set_num_threads(2)
    seed_everything(0)
    config = small_config(projector, scorer)
    model = DynamicAffinityModel(config)
    records = [{"complex_id": "a", "trajectories": [trajectory(4), trajectory(3)]},
               {"complex_id": "b", "trajectories": [trajectory(2)]}]
    output, aux = model(records, return_aux=True)
    assert output.shape == (2,)
    loss, terms = affinity_objective(output, torch.tensor([1., -1.]), aux, config)
    loss.backward()
    assert model.patches.graph.element.weight.grad.abs().sum() > 0
    assert model.temporal.gru.weight_ih_l0.grad.abs().sum() > 0
    if scorer == "quantum": assert model.scoring_features.circuit.weights.grad.abs().sum() > 0
    if projector != "mlp":
        assert len(aux["latent"]) == (4 + 3 + 2) * 3
        assert model.projection.autoencoder.decoder[-1].weight.grad.abs().sum() > 0
    model.eval()
    restored = DynamicAffinityModel(config).eval()
    restored.restore(model.checkpoint_state())
    with torch.no_grad(): assert torch.allclose(model(records), restored(records), atol=1e-6)


def test_temporal_order_sensitivity_and_mean_control():
    seed_everything(1)
    frames = torch.randn(6, 10)
    times = torch.tensor([0., 1., 2., 4., 5., 8.], dtype=torch.float64)
    temporal = TemporalPooling(10, 8, "gru").eval()
    assert not torch.allclose(temporal(frames, times)[0], temporal(frames.flip(0), times)[0], atol=1e-5)
    temporal = TemporalPooling(10, 8, "mean").eval()
    assert torch.allclose(temporal(frames, times)[0], temporal(frames.flip(0), times)[0], atol=1e-6)


def test_graph_rigid_motion_invariance():
    seed_everything(2)
    encoder = InteractionGraphEncoder(12, layers=1).eval()
    tr = trajectory()
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    a = encoder(tr["positions"][0], tr["atomic_numbers"], tr["ligand_mask"])
    b = encoder(tr["positions"][0] @ rotation + 5, tr["atomic_numbers"], tr["ligand_mask"])
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-5)


def test_ligand_dynamics_change_prediction_and_reference_stays_fixed():
    seed_everything(3)
    config = small_config("mlp", "classical")
    model = DynamicAffinityModel(config).eval()
    tr = trajectory()
    moved = {**tr, "positions": tr["positions"].clone()}
    moved["positions"][1:, tr["ligand_mask"]] += 10
    with torch.no_grad():
        a = model([{"complex_id": "a", "trajectories": [tr]}])
        b = model([{"complex_id": "a", "trajectories": [moved]}])
    assert not torch.allclose(a, b, atol=1e-6)
    _, _, anchors_a = model.patches(tr)
    _, _, anchors_b = model.patches(moved)
    assert torch.equal(anchors_a, anchors_b)


def test_replicas_are_aggregated_per_complex():
    seed_everything(4)
    model = DynamicAffinityModel(small_config("mlp", "classical")).eval()
    tr = trajectory()
    with torch.no_grad():
        a = model([{"complex_id": "a", "trajectories": [tr]}])
        b = model([{"complex_id": "a", "trajectories": [tr, tr]}])
    assert torch.allclose(a, b, atol=1e-6)


def test_temporal_validation_rejects_static_or_invalid_data(tmp_path):
    positions = np.random.default_rng(0).normal(size=(4, 5, 3))
    mask = np.array([False] * 3 + [True] * 2)
    path = tmp_path / "tr.npz"
    save_canonical(path, positions, [6] * 5, mask, np.arange(4))
    assert read_trajectory(path, max_frames=3)["positions"].shape == (3, 5, 3)
    with np.load(path, allow_pickle=False) as f: data = {k: f[k] for k in f.files}
    invalid = {**data, "positions": positions[:1]}
    with pytest.raises(ValueError, match="static"): validate_trajectory(invalid)
    invalid = {**data, "time": np.array([0., 1., 1., 2.])}
    with pytest.raises(ValueError, match="increasing"): validate_trajectory(invalid)
    invalid = {**data, "ligand_mask": np.zeros(5, dtype=bool)}
    with pytest.raises(ValueError, match="ligand_mask"): validate_trajectory(invalid)


@pytest.fixture
def complex_manifest(tmp_path):
    rows = []
    for i in range(24):
        for replica in range(2):
            path = tmp_path / f"{i}_{replica}.npz"
            save_canonical(path, np.ones((3, 5, 3)) * i, [6] * 5, [False] * 3 + [True] * 2, np.arange(3))
            rows.append({"complex_id": f"c{i}", "trajectory_id": f"c{i}_r{replica}", "protein_id": f"p{i // 4}",
                         "ligand_id": f"l{i % 4}", "protein_group": f"p{i // 4}", "ligand_group": f"l{i % 4}",
                         "trajectory_path": path.name, "affinity": float(i), "target_type": "pKd", "source": "SYNTHETIC_TEST"})
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest


@pytest.mark.parametrize("method", ["random", "cold_protein", "cold_ligand", "cold_both"])
def test_complex_split_never_separates_replicas(complex_manifest, method):
    frame = read_manifest(complex_manifest)
    assert len(frame) == 24 and all(len(r) == 2 for r in frame.trajectories)
    splits = split_complexes(frame, method, 0, [.5, .25, .25])
    sets = [set(f.complex_id) for f in splits.values()]
    for i in range(3):
        for j in range(i + 1, 3): assert sets[i].isdisjoint(sets[j])


def test_refuse_ambiguous_cold_ligand_and_mixed_endpoints(complex_manifest):
    frame = read_manifest(complex_manifest)
    frame["ligand_identity_verified"] = False
    with pytest.raises(ValueError, match="verified"): split_complexes(frame, "cold_ligand")
    frame["protein_identity_verified"] = False
    with pytest.raises(ValueError, match="verified"): split_complexes(frame, "cold_protein")
    raw = pd.read_csv(complex_manifest)
    raw.loc[0, "target_type"] = "pKi"
    raw.to_csv(complex_manifest, index=False)
    with pytest.raises(ValueError, match="endpoint"): read_manifest(complex_manifest)


def test_fixed_atom_subset_preserves_ligand_motion():
    xyz = np.zeros((3, 6, 3)); xyz[:, :, 0] = np.arange(6)
    xyz[-1, -1, 0] = .1
    selection, info = crop_trajectory(xyz, np.array([6] * 6), np.array([False] * 5 + [True]), 3)
    assert selection[-1] == 5 and len(selection) == 4
    assert info["ligand_atoms"] == 1


def test_mean_full_model_ignores_frame_order_with_fixed_reference():
    seed_everything(5)
    model = DynamicAffinityModel(small_config("mlp", "classical", "mean")).eval()
    tr = trajectory()
    reversed_tr = {**tr, "positions": tr["positions"].flip(0)}
    with torch.no_grad():
        a = model([{"complex_id": "a", "trajectories": [tr]}])
        b = model([{"complex_id": "a", "trajectories": [reversed_tr]}])
    assert torch.allclose(a, b, atol=1e-6)


def test_optional_language_context_with_real_tiny_transformers(small_backbones):
    protein_path, ligand_path = small_backbones
    config = small_config("mlp", "classical")
    config["dynamic"]["language_context"] = True
    config["encoders"]["protein"].update(model=protein_path, cache=None, chunk_size=4)
    config["encoders"]["ligand"].update(model=ligand_path, cache=None, max_length=16)
    model = DynamicAffinityModel(config).eval()
    record = {"complex_id": "a", "trajectories": [trajectory()], "sequence": "ACGAAC", "smiles": "CCO"}
    assert model([record]).shape == (1,)
    assert not any(key.startswith("protein_encoder.backbone.") for key in model.checkpoint_state())
    with pytest.raises(ValueError, match="true sequence"):
        model([{**record, "sequence": ""}])


def test_native_dcd_import_with_explicit_elements_and_pbc(tmp_path):
    import MDAnalysis as mda
    from scripts.export_md_trajectory import export
    # Synthetic I/O fixture, NOT claimed to be a real MD experiment.
    universe = mda.Universe.empty(5, n_residues=2, atom_resindex=[0, 0, 0, 1, 1], trajectory=True)
    for name, values in [("names", ["C1", "N1", "O1", "C2", "O2"]),
                         ("elements", ["C", "N", "O", "C", "O"]),
                         ("resnames", ["ALA", "LIG"]), ("resids", [1, 2]),
                         ("bonds", [(0, 1), (1, 2), (3, 4)])]:
        universe.add_TopologyAttr(name, values)
    coordinates = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [21., 1., 0.], [22., 1., 0.]], dtype=np.float32)
    universe.atoms.positions = coordinates
    universe.dimensions = [20., 20., 20., 90., 90., 90.]
    pdb, dcd = tmp_path / "top.pdb", tmp_path / "frames.dcd"
    universe.atoms.write(str(pdb))
    with mda.Writer(str(dcd), n_atoms=5) as writer:
        for i in range(4):
            xyz = coordinates.copy(); xyz[3:, 1] += i * .1
            universe.atoms.positions = xyz
            writer.write(universe.atoms)
    output = tmp_path / "trajectory.npz"
    export(pdb, dcd, output, "resname ALA", "resname LIG", max_frames=4, max_protein_atoms=3, unwrap=True)
    result = read_trajectory(output)
    assert result["positions"].shape == (4, 5, 3)
    assert result["time_unit"] == "frame_index"
    p, l = result["positions"][:, :3], result["positions"][:, 3:]
    assert torch.cdist(p, l).min() < 5.
