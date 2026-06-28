"""Tests for continuous-signal-first preprocessing."""

import numpy as np

from wesad_arousal.preprocessing import (
    denoise_eda_continuous,
    filter_ecg_continuous,
    filter_respiration_continuous,
)


def test_filters_run_on_long_continuous_signal():
    fs = 700
    n = int(60 * fs)
    t = np.arange(n) / fs
    ecg = np.sin(2 * np.pi * 1.2 * t)
    eda = 2.0 + 0.05 * np.random.randn(n)
    resp = np.sin(2 * np.pi * 0.25 * t)

    ecg_f = filter_ecg_continuous(ecg, fs)
    eda_f = denoise_eda_continuous(eda, fs)
    resp_f = filter_respiration_continuous(resp, fs)

    assert ecg_f.shape == ecg.shape
    assert eda_f.shape == eda.shape
    assert resp_f.shape == resp.shape
    assert not np.allclose(ecg, ecg_f)
