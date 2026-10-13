"""Offline end-to-end optimization test using synthetic tensors, not scientific data."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.model import AffinityModel
from src.utils import load_config, seed_everything, atomic_save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["classical", "quantum"], required=True)
    p.add_argument("--steps", type=int, default=12)
    args = p.parse_args()
    torch.set_num_threads(2)
    seed_everything(0)
    config = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    config["model"].update(patch_dim=16, projection_hidden=12, dropout=0., head_hidden=[16])
    config["patches"].update(window=4, max_patches=8, k=3)
    model = AffinityModel(12, 10, config, args.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    residues = [torch.randn(n, 12) for n in (8, 13, 20, 6)]
    ligands = torch.randn(4, 10)
    target = torch.tensor([0.5, -1., 1., -0.5])
    losses = []
    before = model.projection.network[0].weight.detach().clone()
    for step in range(args.steps):
        output = model(residues, ligands)
        loss = torch.nn.functional.mse_loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        assert torch.isfinite(model.projection.network[0].weight.grad).all()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], losses
    assert not torch.equal(before, model.projection.network[0].weight)
    model.eval()
    with torch.no_grad():
        expected, attention = model(residues, ligands, return_attention=True)
    assert torch.allclose(attention.sum(1), torch.ones(4))
    path = Path("runs/smoke") / args.model / "model.pt"
    atomic_save(model.state_dict(), path)
    restored = AffinityModel(12, 10, config, args.model).eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    with torch.no_grad():
        assert torch.allclose(expected, restored(residues, ligands))
    print(f"PASS {args.model}: MSE {losses[0]:.6f} -> {losses[-1]:.6f}; gradients, pooling, checkpoint verified")


if __name__ == "__main__":
    main()
