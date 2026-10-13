"""Offline check of SAE/Dense AE integration, both scorers and checkpoint roundtrips."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.model import AffinityModel
from src.sae_losses import affinity_objective
from src.utils import load_config, seed_everything, atomic_save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--projector", choices=["sae", "dense_ae"], default="sae")
    p.add_argument("--model", choices=["classical", "quantum"], default="quantum")
    p.add_argument("--sparsity-mode", choices=["l1", "topk"], default="l1")
    args = p.parse_args()
    torch.set_num_threads(2)
    seed_everything(0)
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    config["model"].update(projector=args.projector, patch_dim=16, projection_hidden=12,
                            dropout=0., head_hidden=[16])
    config["patches"].update(window=4, max_patches=8, k=3)
    config["sae"].update(latent_dim=8, hidden_dim=12, sparsity_mode=args.sparsity_mode, topk=3)
    model = AffinityModel(12, 10, config, args.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    residues = [torch.randn(n, 12) for n in (8, 13, 20, 6)]
    ligands, target = torch.randn(4, 10), torch.tensor([.5, -1., 1., -.5])
    losses = []
    for _ in range(20):
        prediction, aux = model(residues, ligands, return_aux=True)
        total, terms = affinity_objective(prediction, target, aux, config)
        optimizer.zero_grad()
        total.backward()
        assert model.projection.autoencoder.decoder[-1].weight.grad is not None
        if args.model == "quantum":
            assert model.scoring_features.circuit.weights.grad is not None
        optimizer.step()
        losses.append(total.item())
    assert losses[-1] < losses[0], losses
    if args.projector == "sae" and args.sparsity_mode == "topk":
        assert (aux["latent"].ne(0).sum(1) <= 3).all()
    model.eval()
    with torch.no_grad():
        expected = model(residues, ligands)
    path = Path("runs/smoke_sae") / args.projector / args.sparsity_mode / args.model / "model.pt"
    atomic_save(model.state_dict(), path)
    restored = AffinityModel(12, 10, config, args.model).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        assert torch.allclose(expected, restored(residues, ligands))
    print(f"PASS {args.projector}/{args.sparsity_mode}/{args.model}: total loss {losses[0]:.6f} -> {losses[-1]:.6f}")


if __name__ == "__main__":
    main()
