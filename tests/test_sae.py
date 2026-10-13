import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.model import AffinityModel
from src.sparse_autoencoder import SparseAutoencoder, DenseAutoencoder
from src.sae_losses import sparsity_loss, reconstruction_loss, affinity_objective
from src.train_utils import LatentStats, feature_signature, verify_transfer, experiment_signature
from src.utils import load_config, seed_everything


def small_config(projector="sae", mode="l1"):
    config = load_config("config.yaml")
    config["model"].update(projector=projector, patch_dim=16, projection_hidden=8,
                            dropout=0., head_hidden=[8])
    config["patches"].update(k=4, window=2)
    config["sae"].update(latent_dim=8, hidden_dim=12, sparsity_mode=mode, topk=3)
    return config


@pytest.mark.parametrize("mode", ["l1", "topk"])
def test_sparse_encoder_losses_and_gradients(mode):
    seed_everything(0)
    model = SparseAutoencoder(16, 8, 12, mode, 3)
    x = torch.randn(10, 16)
    z, reconstructed = model(x)
    assert z.shape == (10, 8) and reconstructed.shape == x.shape
    if mode == "topk":
        assert (z.ne(0).sum(1) <= 3).all()
        assert sparsity_loss(z, mode) == 0
    else:
        assert torch.equal(sparsity_loss(z, mode), z.abs().sum(1).mean())
    (reconstruction_loss(reconstructed, x) + .001 * sparsity_loss(z, mode)).backward()
    assert model.encoder[0].weight.grad.abs().sum() > 0
    assert model.decoder[-1].weight.grad.abs().sum() > 0


def test_topk_ties_and_invalid_dimensions():
    model = SparseAutoencoder(4, 4, 6, "topk", 2)
    for p in model.parameters():
        p.data.zero_()
    model.encoder[-2].bias.data.fill_(1.)
    z = model.encode(torch.zeros(3, 4))
    assert z.tolist() == [[1., 1., 0., 0.]] * 3
    with pytest.raises(ValueError, match="Top-K"):
        SparseAutoencoder(4, 4, 6, "topk", 5)


@pytest.mark.parametrize("variant", ["classical", "quantum"])
def test_mlp_matches_original_golden_outputs(variant):
    torch.set_num_threads(2)
    fixture = json.loads(Path("tests/fixtures/baseline_predictions.json").read_text())
    seed_everything(fixture["seed"])
    model = AffinityModel(6, 5, fixture["config"], variant).eval()
    residues = [torch.randn(3, 6), torch.randn(10, 6)]
    ligands = torch.randn(2, 5)
    with torch.no_grad():
        output, attention = model(residues, ligands, return_attention=True)
    assert torch.allclose(output, torch.tensor(fixture["cases"][variant]["prediction"]), atol=1e-6, rtol=1e-6)
    assert torch.allclose(attention, torch.tensor(fixture["cases"][variant]["attention"]), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("variant", ["classical", "quantum"])
def test_ablation_initialization_is_shared(variant):
    torch.set_num_threads(2)
    seed_everything(0)
    reference = AffinityModel(6, 5, small_config("mlp"), variant).state_dict()
    for projector, mode in [("sae", "l1"), ("sae", "topk"), ("dense_ae", "l1")]:
        seed_everything(0)
        state = AffinityModel(6, 5, small_config(projector, mode), variant).state_dict()
        for key in reference:
            if key.startswith(("patches.", "scoring_features.", "score.", "head.")):
                assert torch.equal(reference[key], state[key]), key


@pytest.mark.parametrize("projector,mode,variant", [
    ("sae", "l1", "classical"), ("sae", "l1", "quantum"),
    ("sae", "topk", "quantum"), ("dense_ae", "l1", "quantum")])
def test_auxiliary_loss_excludes_padding_and_propagates(projector, mode, variant):
    torch.set_num_threads(2)
    seed_everything(0)
    config = small_config(projector, mode)
    model = AffinityModel(6, 5, config, variant)
    residues = [torch.randn(3, 6), torch.randn(10, 6)]
    prediction, aux = model(residues, torch.randn(2, 5), return_aux=True)
    assert len(aux["latent"]) == 2 + 4  # Two valid patches + four, not eight padded patches.
    assert aux["patch_counts"].tolist() == [2, 4]
    loss, parts = affinity_objective(prediction, torch.randn(2), aux, config)
    loss.backward()
    assert model.projection.autoencoder.encoder[0].weight.grad.abs().sum() > 0
    assert model.projection.autoencoder.decoder[-1].weight.grad.abs().sum() > 0
    if variant == "quantum":
        assert model.scoring_features.circuit.weights.grad.abs().sum() > 0
    if projector == "dense_ae" or mode == "topk":
        assert parts["sparsity"] == 0


def test_identity_latent_projection():
    config = small_config()
    config["sae"]["latent_dim"] = config["model"]["n_qubits"]
    model = AffinityModel(6, 5, config)
    assert isinstance(model.projection.to_qubits, torch.nn.Identity)


def test_streaming_latent_stats_and_export(tmp_path):
    stats = LatentStats(topk=2, export=True)
    z = torch.tensor([[0., 1., 2.], [0., 0., 3.], [1., 0., 0.]])
    aux = {"latent": z, "patch_counts": torch.tensor([1, 2]),
           "attention": torch.tensor([[1., 0.], [.4, .6]])}
    records = [{"row_id": 4, "protein_id": "p", "ligand_id": "a"},
               {"row_id": 7, "protein_id": "q", "ligand_id": "b"}]
    stats.update(aux, torch.tensor(.5), torch.tensor(2.), records)
    result = stats.metrics()
    assert result["latent_fraction_zeros"] == 5 / 9
    assert result["latent_mean_active"] == pytest.approx(4 / 3)
    assert result["latent_mean_l1_norm"] == 7 / 3
    assert result["latent_variance"] == pytest.approx(z.var(0, unbiased=False).mean().item())
    path = tmp_path / "latents.npz"
    stats.save(path)
    with np.load(path, allow_pickle=False) as archive:
        assert archive["row_id"].tolist() == [4, 7, 7]
        assert archive["patch_index"].tolist() == [0, 0, 1]
        assert np.array_equal(archive["z"], z.numpy())


def test_pretrained_provenance_rejects_leakage():
    config = small_config()
    context = {"dataset": "davis", "split": "cold_both", "seed": 0,
               "fingerprint": "data-hash", "training_rows": [1, 2, 3]}
    checkpoint = {**context, "config": config, "feature_signature": feature_signature(config)}
    verify_transfer(checkpoint, config, context)
    for key, value in [("seed", 1), ("split", "random"), ("training_rows", [1, 2, 4])]:
        wrong = {**checkpoint, key: value}
        with pytest.raises(ValueError):
            verify_transfer(wrong, config, context)


def test_experiment_signature_ignores_seed_but_checks_loss_weights():
    config = small_config()
    changed = copy.deepcopy(config)
    changed["seed"] = 4
    changed["training"]["output"] = "elsewhere"
    assert experiment_signature(changed) == experiment_signature(config)
    changed["sae"]["sparsity_weight"] *= 2
    assert experiment_signature(changed) != experiment_signature(config)


def test_frozen_autoencoder_still_allows_affinity_gradients():
    config = small_config()
    config["sae"]["frozen"] = True
    model = AffinityModel(6, 5, config, "quantum")
    prediction = model([torch.randn(5, 6)], torch.randn(1, 5))
    prediction.square().sum().backward()
    assert all(p.grad is None for p in model.projection.autoencoder.parameters())
    assert model.projection.to_qubits.weight.grad.abs().sum() > 0
    assert model.patches.encoder[1].weight.grad.abs().sum() > 0


def test_aggregation_rejects_different_loss_configs(tmp_path):
    import subprocess
    import sys
    from src.utils import save_json
    base = {"dataset": "davis", "model": "quantum", "split": "random", "projector": "sae",
            "sparsity_mode": "l1", "training_protocol": "end_to_end", "rmse": 1., "mae": .5,
            "pearson": .1, "spearman": .2, "ci": .6}
    save_json({**base, "seed": 0, "experiment_signature": "weights-a"}, tmp_path / "seed_0/metrics.json")
    save_json({**base, "seed": 1, "experiment_signature": "weights-b"}, tmp_path / "seed_1/metrics.json")
    result = subprocess.run([sys.executable, "aggregate.py", "--root", str(tmp_path)], capture_output=True)
    assert result.returncode != 0 and b"Do not mix" in result.stderr
