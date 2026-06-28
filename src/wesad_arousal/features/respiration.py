"""Respiration and BVP features."""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal

from wesad_arousal.preprocessing import safe_stat


def extract_respiration_features(resp: np.ndarray, fs: float, prefix: str = "resp") -> dict[str, float]:
    signal = np.asarray(resp, dtype=float).reshape(-1)
    if signal.size < fs * 5:
        return {
            f"{prefix}_rate_mean": 0.0,
            f"{prefix}_rate_std": 0.0,
            f"{prefix}_amplitude_mean": safe_stat(signal, np.mean),
            f"{prefix}_amplitude_std": safe_stat(signal, np.std),
            f"{prefix}_plausibility": 0.0,
        }

    rate = _estimate_breathing_rate(signal, fs)
    plausibility = 1.0 if 6.0 <= rate <= 40.0 else 0.0
    return {
        f"{prefix}_rate_mean": rate,
        f"{prefix}_rate_std": safe_stat(signal, np.std),
        f"{prefix}_amplitude_mean": safe_stat(signal, np.mean),
        f"{prefix}_amplitude_std": safe_stat(signal, np.std),
        f"{prefix}_plausibility": plausibility,
    }


def extract_bvp_features(bvp: np.ndarray, fs: float, prefix: str = "bvp") -> dict[str, float]:
    signal = np.asarray(bvp, dtype=float).reshape(-1)
    if signal.size < fs * 3:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_pulse_rate": 0.0,
            f"{prefix}_pulse_variability": 0.0,
        }

    try:
        import neurokit2 as nk

        cleaned = nk.ppg_clean(signal, sampling_rate=fs)
        _, peaks = nk.ppg_peaks(cleaned, sampling_rate=fs)
        peak_idx = np.asarray(peaks.get("PPG_Peaks", []), dtype=int)
        peak_idx = peak_idx[peak_idx > 0]
        if peak_idx.size >= 2:
            ibi = np.diff(peak_idx) / fs
            hr = 60.0 / ibi
            pulse_rate = safe_stat(hr, np.mean)
            pulse_var = safe_stat(hr, np.std)
        else:
            pulse_rate = 0.0
            pulse_var = 0.0
    except Exception:
        pulse_rate = _estimate_breathing_rate(signal, fs)
        pulse_var = safe_stat(signal, np.std)

    return {
        f"{prefix}_mean": safe_stat(signal, np.mean),
        f"{prefix}_std": safe_stat(signal, np.std),
        f"{prefix}_pulse_rate": pulse_rate,
        f"{prefix}_pulse_variability": pulse_var,
    }


def _estimate_breathing_rate(signal: np.ndarray, fs: float) -> float:
    freqs, power = scipy_signal.welch(signal, fs=fs, nperseg=min(len(signal), int(fs * 30)))
    band = (freqs >= 0.1) & (freqs <= 0.7)
    if not np.any(band):
        return 0.0
    peak_freq = freqs[band][np.argmax(power[band])]
    return float(peak_freq * 60.0)
