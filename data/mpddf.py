"""
MPD-DF (Multi-Physiological-Data-based Driver Fatigue) dataset loader.

Reference: Li et al., 2026. "MPD-DF: A multi-physiological-data-based
driver fatigue dataset."

Data: 50 subjects, 32-channel EEG, 500 Hz, ~2h simulated driving.
Labels: Physician-annotated fatigue levels (5 ordered levels),
        mapped to continuous [0, 1] for regression.

Expected directory structure (pre-processed DE features):
    data_root/
    ├── DE_features/
    │   ├── sub01.mat             # contains 'de_features' (C, T, 5)
    │   └── ...
    └── labels/
        ├── sub01.mat             # contains 'fatigue_level' (T,) in {1,2,3,4,5}
        └── ...

If using raw EEG, place files under data_root/raw/ and run preprocessing.
"""

import glob
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


def map_fatigue_levels(levels: np.ndarray, n_levels: int = 5) -> np.ndarray:
    """
    Map ordered physician fatigue levels to continuous [0, 1].

    Levels 1..n_levels are linearly mapped:
        1 -> 0.0 (alert), n_levels -> 1.0 (severely fatigued)

    Args:
        levels: integer fatigue levels (1-indexed)
        n_levels: total number of discrete levels

    Returns:
        Continuous fatigue scores in [0, 1].
    """
    levels = np.asarray(levels, dtype=np.float32)
    continuous = (levels - 1.0) / max(n_levels - 1.0, 1.0)
    return np.clip(continuous, 0.0, 1.0)


def extract_subject_id_mpddf(filename: str) -> int:
    """
    Extract subject ID from MPD-DF filename.
    Expects format like 'sub01.mat' or 'subject_1.mat'.
    """
    basename = os.path.basename(filename)
    match = re.search(r"sub(?:ject)?[_]?(\d+)", basename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.match(r"(\d+)", basename)
    if match:
        return int(match.group(1))
    return -1


def build_mpddf_file_pairs(data_root: str) -> List[Tuple[str, str]]:
    """
    Match DE feature files with label files by basename.

    Args:
        data_root: root directory containing DE_features/ and labels/

    Returns:
        List of (feature_path, label_path) tuples.
    """
    feat_dir = os.path.join(data_root, "DE_features")
    label_dir = os.path.join(data_root, "labels")

    if not os.path.isdir(feat_dir):
        raise FileNotFoundError(
            f"DE_features directory not found: {feat_dir}\n"
            f"Please place pre-processed DE features under {feat_dir}/ "
            f"or run preprocessing first."
        )

    feat_files = sorted(glob.glob(os.path.join(feat_dir, "*.mat")))
    pairs = []
    for feat_path in feat_files:
        fname = os.path.basename(feat_path)
        label_path = os.path.join(label_dir, fname)
        if os.path.exists(label_path):
            pairs.append((feat_path, label_path))
        else:
            print(f"[WARN] Label file not found: {label_path}")

    print(f"[MPD-DF] Matched {len(pairs)} feature-label pairs.")
    return pairs


def _load_de_from_mat(mat: dict) -> np.ndarray:
    """Extract DE features from .mat file. Returns (C, T, 5)."""
    for key in ["de_features", "de_movingAve", "DE", "features"]:
        if key in mat:
            return np.asarray(mat[key], dtype=np.float32)
    raise KeyError(f"DE features not found in .mat, keys = {list(mat.keys())}")


def _load_label_from_mat(mat: dict) -> np.ndarray:
    """Extract fatigue labels from .mat file. Returns (T,) in [0,1]."""
    for key in ["fatigue_level", "fatigue", "label", "y"]:
        if key in mat:
            arr = np.squeeze(mat[key]).astype(np.float32)
            # If discrete levels (1-5), map to [0,1]
            if arr.max() > 1.0:
                arr = map_fatigue_levels(arr)
            return arr
    raise KeyError(f"Label not found in .mat, keys = {list(mat.keys())}")


class MPDDFDataset(Dataset):
    """
    MPD-DF sequence dataset for streaming fatigue regression.

    Each sample is one full session: (B, C, T) features and (T,) labels.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        normalize: bool = True,
        cache: bool = True,
        augment: bool = False,
        channel_indices: Optional[List[int]] = None,
    ):
        super().__init__()
        self.pairs = pairs
        self.normalize = normalize
        self.cache = cache
        self.augment = augment
        self.channel_indices = channel_indices

        self._cache = []
        if self.cache:
            print("[MPD-DF] Caching data into memory...")
            for feat_path, label_path in self.pairs:
                self._cache.append(self._load_pair(feat_path, label_path))
            print(f"[MPD-DF] Cached {len(self._cache)} samples.")

    def _load_pair(self, feat_path: str, label_path: str):
        feat_mat = sio.loadmat(feat_path)
        label_mat = sio.loadmat(label_path)

        de = _load_de_from_mat(feat_mat)          # (C, T, B)
        label = _load_label_from_mat(label_mat)    # (T,)

        x = np.transpose(de, (2, 0, 1)).astype(np.float32)  # (B, C, T)
        if self.channel_indices is not None:
            x = x[:, self.channel_indices, :]

        # Align lengths
        T = min(x.shape[2], len(label))
        x = x[:, :, :T]
        label = label[:T]

        return x, label

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.cache:
            x_np, y_np = self._cache[idx]
            x = torch.from_numpy(x_np).clone()
            y = torch.tensor(y_np, dtype=torch.float32).clone()
        else:
            feat_path, label_path = self.pairs[idx]
            x_np, y_np = self._load_pair(feat_path, label_path)
            x = torch.from_numpy(x_np)
            y = torch.tensor(y_np, dtype=torch.float32)

        if self.normalize:
            mean = x.mean(dim=2, keepdim=True)
            std = x.std(dim=2, keepdim=True, unbiased=False).clamp(min=1e-5)
            x = (x - mean) / std

        if self.augment:
            noise = torch.randn_like(x) * 0.02
            x = x + noise

        return x, y


__all__ = [
    "MPDDFDataset",
    "build_mpddf_file_pairs",
    "extract_subject_id_mpddf",
    "map_fatigue_levels",
]
