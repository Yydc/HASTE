"""
SADT (Sustained-Attention Driving Task) dataset loader.

Reference: Cao et al., 2019. "Multi-channel EEG recordings during a
sustained-attention driving task."

Data: 27 subjects, 62 sessions, 30 scalp EEG channels (+ 2 mastoid refs),
      500 Hz sampling rate, ~90 min sustained-attention driving.
Labels: Reaction time (RT) from lane-departure events, converted to
        a continuous Drowsiness Index (DI) in [0, 1].

Expected directory structure (pre-processed DE features):
    data_root/
    ├── DE_features/
    │   ├── sub01_sess01.mat      # contains 'de_features' (C, T, 5)
    │   ├── sub01_sess02.mat
    │   └── ...
    └── labels/
        ├── sub01_sess01.mat      # contains 'drowsiness_index' (T,)
        └── ...

If using raw EEG, place .set/.mat files under data_root/raw/ and run
preprocessing with --preprocess flag.
"""

import glob
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


def rt_to_drowsiness_index(rt: np.ndarray, rt_min: float = 0.3,
                           rt_max: float = 3.0) -> np.ndarray:
    """
    Convert reaction time (seconds) to drowsiness index in [0, 1].

    Uses a monotonic, long-tail-robust mapping:
        DI = clip((log(RT) - log(rt_min)) / (log(rt_max) - log(rt_min)), 0, 1)

    Larger RT -> higher drowsiness (closer to 1).

    Args:
        rt: reaction time values in seconds
        rt_min: RT corresponding to full alertness (DI=0)
        rt_max: RT corresponding to severe drowsiness (DI=1)

    Returns:
        Drowsiness index in [0, 1].
    """
    log_rt = np.log(np.clip(rt, 1e-3, None))
    log_min = np.log(rt_min)
    log_max = np.log(rt_max)
    di = (log_rt - log_min) / (log_max - log_min + 1e-8)
    return np.clip(di, 0.0, 1.0).astype(np.float32)


def extract_subject_id_sadt(filename: str) -> int:
    """
    Extract subject ID from SADT filename.
    Expects format like 'sub01_sess01.mat' or 'subject_1_session_2.mat'.
    """
    basename = os.path.basename(filename)
    match = re.search(r"sub(?:ject)?[_]?(\d+)", basename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.match(r"(\d+)_", basename)
    if match:
        return int(match.group(1))
    return -1


def build_sadt_file_pairs(data_root: str) -> List[Tuple[str, str]]:
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

    print(f"[SADT] Matched {len(pairs)} feature-label pairs.")
    return pairs


def _load_de_from_mat(mat: dict) -> np.ndarray:
    """Extract DE features from .mat file. Returns (C, T, 5)."""
    for key in ["de_features", "de_movingAve", "DE", "features"]:
        if key in mat:
            return np.asarray(mat[key], dtype=np.float32)
    raise KeyError(f"DE features not found in .mat, keys = {list(mat.keys())}")


def _load_label_from_mat(mat: dict) -> np.ndarray:
    """Extract drowsiness label from .mat file. Returns (T,)."""
    for key in ["drowsiness_index", "DI", "label", "y", "rt"]:
        if key in mat:
            arr = np.squeeze(mat[key]).astype(np.float32)
            # If raw RT values, convert to DI
            if key == "rt":
                arr = rt_to_drowsiness_index(arr)
            return arr
    raise KeyError(f"Label not found in .mat, keys = {list(mat.keys())}")


class SADTDataset(Dataset):
    """
    SADT sequence dataset for streaming fatigue regression.

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
            print("[SADT] Caching data into memory...")
            for feat_path, label_path in self.pairs:
                self._cache.append(self._load_pair(feat_path, label_path))
            print(f"[SADT] Cached {len(self._cache)} samples.")

    def _load_pair(self, feat_path: str, label_path: str):
        feat_mat = sio.loadmat(feat_path)
        label_mat = sio.loadmat(label_path)

        de = _load_de_from_mat(feat_mat)         # (C, T, B)
        label = _load_label_from_mat(label_mat)   # (T,)

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
    "SADTDataset",
    "build_sadt_file_pairs",
    "extract_subject_id_sadt",
    "rt_to_drowsiness_index",
]
