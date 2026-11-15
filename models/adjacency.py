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


def build_knn_adjacency(coords: np.ndarray, k: int = 5, self_loop: bool = True) -> torch.Tensor:
    """
    Physical graph A^phy in {0,1}^{C x C} (Sec 4.2): kNN rule on electrode
    coordinates, symmetrized, with self-loops. coords: (C, 2 or 3)
    """
    coords = np.asarray(coords, dtype=np.float64)
    C = coords.shape[0]
    if coords.ndim != 2 or not 1 <= k < C:
        raise ValueError(f"need coords of shape (C, D) and 1 <= k < C, got {coords.shape}, k={k}")
    dist2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(dist2, np.inf)  # exclude self from the k neighbours
    mask = np.zeros((C, C), dtype=np.float32)
    for i in range(C):
        mask[i, np.argsort(dist2[i], kind="stable")[:k]] = 1.0
    mask = np.maximum(mask, mask.T)
    if self_loop:
        np.fill_diagonal(mask, 1.0)
    return torch.from_numpy(mask)


def coords_from_channel_names(channel_names) -> np.ndarray:
    """3-D positions of 10-20 channel names (case-insensitive) from MNE's standard_1005 montage."""
    import mne

    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    lut = {name.upper(): xyz for name, xyz in pos.items()}
    missing = [n for n in channel_names if str(n).strip().upper() not in lut]
    if missing:
        raise ValueError(f"Channels not in the standard_1005 montage: {missing}")
    return np.asarray([lut[str(n).strip().upper()] for n in channel_names], dtype=np.float64)


def row_normalize(adj: torch.Tensor) -> torch.Tensor:
    """
    Row-normalize adjacency to form a row-stochastic matrix (Sec 4.2).
    P^phy_{ij} = A_{ij} / sum_k A_{ik}
    """
    row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return adj / row_sum


__all__ = [
    "build_full_adjacency",
    "build_knn_gaussian",
    "build_knn_adjacency",
    "coords_from_channel_names",
    "row_normalize",
]
