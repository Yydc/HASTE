"""
Shared EEG preprocessing utilities for SADT and MPD-DF datasets.

Pipeline: raw EEG -> bandpass 1-50 Hz -> downsample to 128 Hz
          -> 1s non-overlapping windows -> 5-band DE features.

Bands:
    delta (1-3 Hz), theta (4-7 Hz), alpha (8-13 Hz),
    beta (14-30 Hz), gamma (31-50 Hz)
"""

import numpy as np

# Band definitions: (name, low_freq, high_freq)
BANDS = [
    ("delta", 1, 3),
    ("theta", 4, 7),
    ("alpha", 8, 13),
    ("beta", 14, 30),
    ("gamma", 31, 50),
]


def bandpass_filter(data: np.ndarray, sfreq: float,
                    l_freq: float = 1.0, h_freq: float = 50.0) -> np.ndarray:
    """
    Apply bandpass filter to EEG data using MNE.

    Args:
        data: (n_channels, n_samples)
        sfreq: sampling frequency in Hz
        l_freq: low cutoff frequency
        h_freq: high cutoff frequency

    Returns:
        Filtered data with same shape.
    """
    import mne

    info = mne.create_info(
        ch_names=[f"EEG{i}" for i in range(data.shape[0])],
        sfreq=sfreq,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.filter(l_freq, h_freq, verbose=False)
    return raw.get_data()


def resample(data: np.ndarray, sfreq_orig: float,
             sfreq_target: float = 128.0) -> np.ndarray:
    """
    Resample EEG data using MNE.

    Args:
        data: (n_channels, n_samples)
        sfreq_orig: original sampling frequency
        sfreq_target: target sampling frequency

    Returns:
        Resampled data.
    """
    import mne

    info = mne.create_info(
        ch_names=[f"EEG{i}" for i in range(data.shape[0])],
        sfreq=sfreq_orig,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.resample(sfreq_target, verbose=False)
    return raw.get_data()


def compute_de_single_band(segment: np.ndarray, sfreq: float,
                           low: float, high: float) -> np.ndarray:
    """
    Compute differential entropy for one frequency band on a single window.

    DE = 0.5 * log(2 * pi * e * variance_of_bandpassed_signal)

    Args:
        segment: (n_channels, n_samples) one window of EEG
        sfreq: sampling frequency
        low: band low frequency
        high: band high frequency

    Returns:
        DE values: (n_channels,)
    """
    filtered = bandpass_filter(segment, sfreq, low, high)
    var = np.var(filtered, axis=1)
    var = np.clip(var, 1e-10, None)  # numerical stability
    de = 0.5 * np.log(2 * np.pi * np.e * var)
    return de


def compute_de_5bands(data: np.ndarray, sfreq: float = 128.0,
                      window_sec: float = 1.0) -> np.ndarray:
    """
    Compute 5-band DE features from continuous EEG.

    Args:
        data: (n_channels, n_samples) bandpass-filtered, resampled EEG
        sfreq: sampling frequency (default 128 Hz)
        window_sec: window length in seconds (default 1.0)

    Returns:
        DE features: (n_channels, n_windows, 5)
    """
    n_channels, n_samples = data.shape
    win_samples = int(sfreq * window_sec)
    n_windows = n_samples // win_samples

    de_features = np.zeros((n_channels, n_windows, len(BANDS)), dtype=np.float32)

    for w in range(n_windows):
        start = w * win_samples
        end = start + win_samples
        segment = data[:, start:end]

        for b, (name, low, high) in enumerate(BANDS):
            de_features[:, w, b] = compute_de_single_band(segment, sfreq, low, high)

    return de_features


def preprocess_raw_to_de(
    raw_data: np.ndarray,
    sfreq: float = 500.0,
    target_sfreq: float = 128.0,
    l_freq: float = 1.0,
    h_freq: float = 50.0,
    window_sec: float = 1.0,
) -> np.ndarray:
    """
    Full preprocessing pipeline: raw EEG -> 5-band DE features.

    Args:
        raw_data: (n_channels, n_samples) raw EEG data
        sfreq: original sampling frequency
        target_sfreq: target sampling frequency after resampling
        l_freq: bandpass low cutoff
        h_freq: bandpass high cutoff
        window_sec: DE window length in seconds

    Returns:
        DE features: (n_channels, n_windows, 5)
    """
    # Step 1: Bandpass filter
    data = bandpass_filter(raw_data, sfreq, l_freq, h_freq)

    # Step 2: Downsample
    if sfreq != target_sfreq:
        data = resample(data, sfreq, target_sfreq)

    # Step 3: Compute 5-band DE
    de = compute_de_5bands(data, target_sfreq, window_sec)

    return de


__all__ = [
    "BANDS",
    "bandpass_filter",
    "resample",
    "compute_de_single_band",
    "compute_de_5bands",
    "preprocess_raw_to_de",
]
