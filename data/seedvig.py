import glob
import os
import re
from typing import List, Tuple, Optional

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


def extract_de_5bands_from_mat(mat: dict) -> np.ndarray:
    """
    Extract de_movingAve (17 x 885 x 5).
    Supports:
      - root field 'de_movingAve'
      - struct 'EEG_Feature_5Bands' with field 'de_movingAve'
    """
    if "EEG_Feature_5Bands" in mat:
        struct = mat["EEG_Feature_5Bands"]
        de = struct["de_movingAve"][0, 0]
    elif "de_movingAve" in mat:
        de = mat["de_movingAve"]
    else:
        raise KeyError(
            f"'de_movingAve' or 'EEG_Feature_5Bands' not found in .mat file, keys = {list(mat.keys())}"
        )
    return de  # (C, T, B)


def extract_perclos_from_mat(mat: dict) -> np.ndarray:
    """Extract PERCLOS label with a few common fallbacks."""
    for key in ["PERCLOS", "perclos", "label", "y"]:
        if key in mat:
            arr = np.squeeze(mat[key])
            return arr.astype(np.float32)
    raise KeyError(f"PERCLOS label not found, keys = {list(mat.keys())}")


def extract_subject_id(filename: str) -> int:
    """
    Extract subject ID from SEED-VIG filename.
    Assumes format like '1_20151124_...' where first number is subject ID.
    Returns -1 if parsing fails.
    """
    basename = os.path.basename(filename)
    match = re.match(r"(\d+)_", basename)
    if match:
        return int(match.group(1))
    return -1


def build_file_pairs(eeg_dir: str, perclos_dir: str) -> List[Tuple[str, str]]:
    """
    Match EEG and PERCLOS files by identical basenames.
    """
    eeg_files = sorted(glob.glob(os.path.join(eeg_dir, "*.mat")))
    pairs = []
    for eeg_path in eeg_files:
        fname = os.path.basename(eeg_path)
        perclos_path = os.path.join(perclos_dir, fname)
        if os.path.exists(perclos_path):
            pairs.append((eeg_path, perclos_path))
        else:
            print(f"[WARN] Label file not found: {perclos_path}")
    print(f"Matched {len(pairs)} EEG-PERCLOS pairs.")
    return pairs


class Augmenter:
    """Simple EEG feature augmentation with noise and band dropout."""

    def __init__(self, noise_std=0.05, band_drop_prob=0.2):
        self.noise_std = noise_std
        self.band_drop_prob = band_drop_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to input tensor.
        x shape: (Bands, Channels, Time) = (5, 17, 885)
        """
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        if self.band_drop_prob > 0 and torch.rand(1).item() < self.band_drop_prob:
            num_bands = x.shape[0]
            drop_idx = torch.randint(0, num_bands, (1,)).item()
            x[drop_idx] = 0.0

        return x


class SEEDVIGSequenceDataset(Dataset):
    """
    Each sample: one full 885-step sequence with 5-band DE features and PERCLOS sequence labels.
    Labels are kept in [0,1] range as per competition requirements.
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

        self.augmenter = Augmenter(noise_std=0.02, band_drop_prob=0.1)

        self._cache = []
        if self.cache:
            print("Caching data into memory...")
            for eeg_path, perclos_path in self.pairs:
                self._cache.append(self._load_pair(eeg_path, perclos_path))
            print(f"SEEDVIGSequenceDataset: Cached {len(self._cache)} samples.")

    def _load_pair(self, eeg_path: str, perclos_path: str):
        eeg_mat = sio.loadmat(eeg_path)
        perclos_mat = sio.loadmat(perclos_path)

        de = extract_de_5bands_from_mat(eeg_mat)          # (C, T, B)
        perclos = extract_perclos_from_mat(perclos_mat)   # (T,)

        x = np.transpose(de, (2, 0, 1)).astype(np.float32)  # (B, C, T)
        if self.channel_indices is not None:
            x = x[:, self.channel_indices, :]
        y = perclos.astype(np.float32)  # (T,) in [0,1] range for competition
        return x, y

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.cache:
            x_np, y_np = self._cache[idx]
            x = torch.from_numpy(x_np).clone()
            y = torch.tensor(y_np, dtype=torch.float32).clone()
        else:
            eeg_path, perclos_path = self.pairs[idx]
            x_np, y_np = self._load_pair(eeg_path, perclos_path)
            x = torch.from_numpy(x_np)
            y = torch.tensor(y_np, dtype=torch.float32)

        # channel selection is already applied in _load_pair when caching.
        # If cache is disabled and channel_indices provided, ensure selection here too (defensive).
        if self.channel_indices is not None and not self.cache:
            x = x[:, self.channel_indices, :]

        if self.normalize:
            mean = x.mean(dim=2, keepdim=True)
            std = x.std(dim=2, keepdim=True, unbiased=False)
            std = torch.clamp(std, min=1e-5)
            x = (x - mean) / std

        if self.augment:
            x = self.augmenter(x)

        return x, y


__all__ = [
    "extract_de_5bands_from_mat",
    "extract_perclos_from_mat",
    "build_file_pairs",
    "extract_subject_id",
    "Augmenter",
    "SEEDVIGSequenceDataset",
]
