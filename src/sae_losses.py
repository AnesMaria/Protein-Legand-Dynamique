"""Loss conventions are explicit and share the same normalization in both modes."""
import torch
from torch.nn import functional as F


def reconstruction_loss(reconstructed, target):
    if reconstructed.shape != target.shape or target.ndim != 2 or not len(target):
        raise ValueError("Reconstruction requires nonempty equal-sized [N,D] matrices")
    return F.mse_loss(reconstructed, target.detach())  # Mean over valid patches and input dimensions.


def sparsity_loss(latent, mode):
    if mode not in ("l1", "topk", "none"):
        raise ValueError(f"Unknown sparsity mode: {mode}")
    # L1 norm per patch, averaged over patches (not divided by latent_dim).
    # Top-K enforces sparsity by its mask, without an additional L1 penalty.
    return latent.abs().sum(dim=-1).mean() if mode == "l1" else latent.new_zeros(())


def autoencoder_losses(aux, mode):
    return reconstruction_loss(aux["reconstructed"], aux["patch_target"]), sparsity_loss(aux["latent"], mode)


def affinity_objective(prediction, target, aux, config):
    affinity = F.mse_loss(prediction, target)
    if "latent" not in aux:
        zero = affinity.new_zeros(())
        return affinity, {"affinity": affinity, "reconstruction": zero, "sparsity": zero}
    sae = config["sae"]
    if sae["reconstruction_weight"] < 0 or sae["sparsity_weight"] < 0:
        raise ValueError("Loss weights must be nonnegative")
    mode = "none" if config["model"]["projector"] == "dense_ae" else sae["sparsity_mode"]
    reconstruction, sparsity = autoencoder_losses(aux, mode)
    total = affinity + sae["reconstruction_weight"] * reconstruction + sae["sparsity_weight"] * sparsity
    return total, {"affinity": affinity, "reconstruction": reconstruction, "sparsity": sparsity}
