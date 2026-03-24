import numpy as np
import torch


def build_full_adjacency(num_channels: int) -> torch.Tensor:
    """Fully connected adjacency with self-loops."""
    return torch.ones(num_channels, num_channels, dtype=torch.float32)


def build_knn_gaussian(
    coords: np.ndarray,
    k: int = 4,
    sigma: float = 1.0,
    self_loop: bool = True,
) -> torch.Tensor:
    """
    Build symmetric k-NN adjacency with Gaussian weights from electrode coordinates.
    coords: (C, 2 or 3)
    """
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[0] < 2:
        raise ValueError("coords must be shape (C, D) with C>=2")
    C = coords.shape[0]
    diff = coords[:, None, :] - coords[None, :, :]
    dist2 = (diff ** 2).sum(-1)  # (C, C)

    weights = np.exp(-dist2 / (2 * (sigma ** 2)))
    mask = np.zeros_like(weights)
    for i in range(C):
        idx = np.argsort(dist2[i])
        keep = idx[1 : k + 1]  # exclude self
        mask[i, keep] = 1.0
    mask = np.maximum(mask, mask.T)
    if self_loop:
        np.fill_diagonal(mask, 1.0)

    adj = torch.from_numpy(weights * mask).float()
    return adj


def row_normalize(adj: torch.Tensor) -> torch.Tensor:
    """
    Row-normalize adjacency to form a row-stochastic matrix (Eq. 10).
    P^phy_{ij} = A_{ij} / sum_k A_{ik}
    """
    row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return adj / row_sum


__all__ = ["build_full_adjacency", "build_knn_gaussian", "row_normalize"]
