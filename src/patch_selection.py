import torch


def farthest_point_sampling(points, k):
    if points.ndim != 2 or len(points) == 0 or k < 1:
        raise ValueError("FPS requires nonempty [N,D] points and k>=1")
    k = min(k, len(points))
    # Translation invariant and deterministic initial center.
    first = (points - points.mean(0)).square().sum(-1).argmax()
    indices = []
    distance = torch.full((len(points),), float("inf"), device=points.device)
    current = first
    for _ in range(k):
        indices.append(current)
        distance = torch.minimum(distance, (points - points[current]).square().sum(-1))
        distance[torch.stack(indices)] = -1  # Never select identical-position nodes twice.
        current = distance.argmax()
    return torch.stack(indices)


def select_patches(patches, k, centers=None):
    if len(patches) == 0 or k < 1:
        raise ValueError("No patches or invalid K")
    if centers is not None:
        indices = farthest_point_sampling(centers.detach(), k)
    else:
        # Uniform sequence coverage; no fabricated 3D coordinates.
        indices = torch.linspace(0, len(patches) - 1, min(k, len(patches)),
                                 device=patches.device).round().long()
    return patches[indices], indices
