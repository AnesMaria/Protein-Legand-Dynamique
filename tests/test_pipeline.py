import itertools

import numpy as np
import pandas as pd
import pytest
import torch

from prepare_data import normalize
from src.datasets import read_table, make_splits
from src.metrics import concordance_index, regression_metrics
from src.model import AffinityModel
from src.patch_encoder import PatchEncoder
from src.patch_selection import farthest_point_sampling
from src.pocket_data import graph_from_pocket
from src.utils import load_config, seed_everything


@pytest.fixture
def frame(tmp_path):
    rows = [{"protein_id": f"p{i}", "ligand_id": f"l{j}", "sequence": "A" * (i + 1),
             "smiles": "C" * (j + 1), "affinity": i + j}
            for i, j in itertools.product(range(12), repeat=2)]
    path = tmp_path / "pairs.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return read_table(path)


@pytest.mark.parametrize("method", ["random", "cold_protein", "cold_ligand", "cold_both"])
def test_splits(frame, method):
    a = make_splits(frame, method, seed=0, fractions=(0.5, 0.25, 0.25))
    b = make_splits(frame, method, seed=0, fractions=(0.5, 0.25, 0.25))
    assert all(a[k].row_id.tolist() == b[k].row_id.tolist() for k in a)
    for x, y in itertools.combinations(a.values(), 2):
        if method in ("cold_protein", "cold_both"):
            assert set(x.sequence).isdisjoint(y.sequence)
        if method in ("cold_ligand", "cold_both"):
            assert set(x.smiles).isdisjoint(y.smiles)
    if method == "cold_both":
        assert sum(map(len, a.values())) < len(frame)
    else:
        assert sum(map(len, a.values())) == len(frame)


def test_ci_against_brute_force():
    rng = np.random.default_rng(0)
    for n in [1, 2, 8, 100]:
        y, p = rng.integers(0, 4, (2, n))
        scores = [1. if (y[i] - y[j]) * (p[i] - p[j]) > 0 else .5 if p[i] == p[j] else 0.
                  for i, j in itertools.combinations(range(n), 2) if y[i] != y[j]]
        expected = np.mean(scores) if scores else None
        assert concordance_index(y, p) == expected
    assert regression_metrics([1, 1], [2, 2])["pearson"] is None


def test_fps_duplicates_and_rotation():
    points = torch.zeros(5, 3)
    assert len(set(farthest_point_sampling(points, 5).tolist())) == 5
    points = torch.randn(12, 3)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    assert torch.equal(farthest_point_sampling(points, 5), farthest_point_sampling(points @ rotation + 4, 5))


@pytest.mark.parametrize("variant", ["classical", "quantum"])
def test_model_gradient_mask_and_permutation(variant):
    torch.set_num_threads(2)
    seed_everything(0)
    config = load_config("config.yaml")
    config["model"].update(patch_dim=16, projection_hidden=8, dropout=0., head_hidden=[8])
    config["patches"].update(k=4, window=2)
    model = AffinityModel(6, 5, config, variant)
    residues = [torch.randn(3, 6), torch.randn(10, 6)]
    ligands = torch.randn(2, 5)
    output, attention = model(residues, ligands, return_attention=True)
    assert (attention[0, 2:] == 0).all()
    assert torch.allclose(attention.sum(1), torch.ones(2))
    output.square().sum().backward()
    assert model.projection.network[0].weight.grad.abs().sum() > 0
    if variant == "quantum":
        assert model.scoring_features.circuit.weights.grad.abs().sum() > 0
    model.eval()
    assert torch.allclose(model(residues[::-1], ligands.flip(0)), output.detach().flip(0), atol=1e-6)


def test_optional_graph_features_and_alignment():
    encoder = PatchEncoder(6, 10, max_patches=5, k=3)
    residues = torch.randn(12, 6)
    graph = {"coords": torch.randn(6, 3), "residue_index": torch.arange(6)}
    assert encoder(residues, graph).shape == (3, 10)
    graph.update(node_s=torch.randn(6, 4), node_v=torch.randn(6, 2, 3),
                 edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
                 edge_s=torch.randn(3, 5), edge_v=torch.randn(3, 1, 3))
    assert torch.isfinite(encoder(residues, graph)).all()
    graph["residue_index"][0] = 20
    with pytest.raises(ValueError, match="alignment"):
        encoder(residues, graph)


def test_affinity_units_and_duplicate_policy():
    raw = pd.DataFrame({"Target_ID": ["a", "b"], "Drug_ID": ["x", "y"],
                        "Target": ["AAA", "AAA"], "Drug": ["CC", "CC"], "Y": [10., 100.]})
    df, _ = normalize(raw, "davis", affinity_unit="nM")
    assert len(df) == 1 and df.affinity.iloc[0] == 7.5
    df, _ = normalize(raw, "bindingdb")
    assert df.affinity.iloc[0] == 100.
    with pytest.raises(ValueError):
        normalize(raw, "kiba", affinity_unit="nM")


def test_pocket_alignment_exact_only():
    lines = []
    for i, aa in enumerate(["ALA", "GLY", "VAL"], start=1):
        lines.append(f"ATOM  {i:5d}  CA  {aa} A{i:4d}    {float(i):8.3f}{0.:8.3f}{0.:8.3f}{1.:6.2f}{90.:6.2f}           C")
    table = pd.DataFrame({"chain": ["A"] * 3, "residue_label": [1, 2, 3], "pocket": [1, 1, 0],
                          "score": [.2, .4, .0], "probability": [.5, .9, .0]})
    graph = graph_from_pocket("\n".join(lines), table, "AGV")
    assert graph["residue_index"].tolist() == [0, 1]
    graph = graph_from_pocket("\n".join(lines), table, "AAV")
    assert "residue_index" not in graph


def test_fingerprint_detects_graph_changes(frame, tmp_path):
    from src.engine import table_fingerprint
    graph_path = tmp_path / "graph.pt"
    torch.save({"coords": torch.zeros(2, 3)}, graph_path)
    frame["graph_path"] = str(graph_path)
    original = table_fingerprint(frame)
    assert original == table_fingerprint(frame.copy())
    torch.save({"coords": torch.ones(2, 3)}, graph_path)
    assert original != table_fingerprint(frame)
