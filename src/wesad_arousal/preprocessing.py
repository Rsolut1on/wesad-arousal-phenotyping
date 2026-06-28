"""Conservative preprocessing helpers.

All bandpass/lowpass operations use zero-phase ``scipy.signal.filtfilt`` (Butterworth)
to avoid phase distortion that would shift R-peak timing or respiratory peaks.

Filtering is applied to the full continuous recording **before** window/segment
extraction to reduce edge artifacts at segment boundaries.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

ECG_BANDPASS_HZ = (0.5, 40.0)
EDA_DENOISE_CUTOFF_HZ = 1.0
RESP_BANDPASS_HZ = (0.05, 0.7)


def bandpass_filter(
    data: np.ndarray,
    fs: float,
    low: float,
    high: float,
    order: int = 3,
) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.size < order * 3:
        return arr
    nyq = 0.5 * fs
    low_norm = max(low / nyq, 1e-4)
    high_norm = min(high / nyq, 0.99)
    if low_norm >= high_norm:
        return arr
    b, a = scipy_signal.butter(order, [low_norm, high_norm], btype="band")
    return scipy_signal.filtfilt(b, a, arr)


def lowpass_filter(data: np.ndarray, fs: float, cutoff: float, order: int = 3) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.size < order * 3:
        return arr
    nyq = 0.5 * fs
    norm = min(cutoff / nyq, 0.99)
    b, a = scipy_signal.butter(order, norm, btype="low")
    return scipy_signal.filtfilt(b, a, arr)


def filter_ecg_continuous(ecg: np.ndarray, fs: float) -> np.ndarray:
    low, high = ECG_BANDPASS_HZ
    return bandpass_filter(ecg, fs, low, high)


def denoise_eda_continuous(eda: np.ndarray, fs: float) -> np.ndarray:
    return lowpass_filter(eda, fs, EDA_DENOISE_CUTOFF_HZ)


def filter_respiration_continuous(resp: np.ndarray, fs: float) -> np.ndarray:
    low, high = RESP_BANDPASS_HZ
    return bandpass_filter(resp, fs, low, high)


def preprocess_continuous_signals(
    signal_arrays: dict[str, np.ndarray],
    cfg: dict,
) -> dict[str, np.ndarray]:
    """Filter full continuous signals once per subject before windowing."""
    fs_chest = cfg["sampling_rates"]["chest"]
    wrist_fs = cfg["sampling_rates"]["wrist"]
    processed: dict[str, np.ndarray] = {}

    if "chest_ECG" in signal_arrays:
        processed["chest_ECG"] = filter_ecg_continuous(signal_arrays["chest_ECG"], fs_chest)
    if "chest_EDA" in signal_arrays:
        processed["chest_EDA"] = denoise_eda_continuous(signal_arrays["chest_EDA"], fs_chest)
    if "chest_Resp" in signal_arrays:
        processed["chest_Resp"] = filter_respiration_continuous(signal_arrays["chest_Resp"], fs_chest)
    if "wrist_EDA" in signal_arrays:
        processed["wrist_EDA"] = denoise_eda_continuous(signal_arrays["wrist_EDA"], wrist_fs["EDA"])

    return processed


def zscore(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    std = np.std(arr)
    if std <= 0:
        return arr - np.mean(arr)
    return (arr - np.mean(arr)) / std


def safe_stat(data: np.ndarray, fn, default: float = 0.0) -> float:
    arr = np.asarray(data, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(fn(arr))
