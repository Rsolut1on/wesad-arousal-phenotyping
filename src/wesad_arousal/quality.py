"""Signal quality control metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from wesad_arousal.labels import map_label_code, valid_label_values


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def compute_signal_qc(
    signal: np.ndarray,
    signal_name: str,
    device: str,
    fs: float,
    valid_range: tuple[float, float] | None = None,
) -> dict[str, float]:
    arr = np.asarray(signal, dtype=float)
    if arr.size == 0:
        return {
            "missingness": 1.0,
            "flatline_fraction": 1.0,
            "out_of_range_fraction": 1.0,
            "jump_fraction": 0.0,
            "coverage_seconds": 0.0,
        }

    finite = np.isfinite(arr)
    missingness = 1.0 - _safe_ratio(np.sum(finite), arr.size)
    arr = arr[finite] if np.any(finite) else np.array([0.0])

    diffs = np.diff(arr) if arr.size > 1 else np.array([0.0])
    flatline_fraction = _safe_ratio(np.sum(np.abs(diffs) < 1e-8), max(len(diffs), 1))

    if valid_range is not None:
        low, high = valid_range
        out_of_range = np.sum((arr < low) | (arr > high))
        out_of_range_fraction = _safe_ratio(out_of_range, arr.size)
    else:
        out_of_range_fraction = 0.0

    if arr.size > 1:
        scale = np.std(arr) if np.std(arr) > 0 else 1.0
        jump_fraction = _safe_ratio(np.sum(np.abs(diffs) > 5 * scale), len(diffs))
    else:
        jump_fraction = 0.0

    return {
        "missingness": missingness,
        "flatline_fraction": flatline_fraction,
        "out_of_range_fraction": out_of_range_fraction,
        "jump_fraction": jump_fraction,
        "coverage_seconds": arr.size / fs if fs > 0 else 0.0,
    }


def qc_row(
    subject_id: str,
    device: str,
    signal_name: str,
    metrics: dict[str, float],
    extra: dict | None = None,
) -> dict:
    row = {
        "subject_id": subject_id,
        "device": device,
        "signal": signal_name,
        **metrics,
    }
    if extra:
        row.update(extra)
    return row


def label_coverage_by_condition(labels: np.ndarray, cfg: dict) -> pd.DataFrame:
    rows = []
    total = len(labels)
    for code in sorted(valid_label_values(cfg)):
        count = int(np.sum(labels == code))
        rows.append(
            {
                "label_code": code,
                "label_name": map_label_code(code, cfg),
                "samples": count,
                "seconds": count / cfg["sampling_rates"]["label"],
                "fraction": count / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def aggregate_subject_qc(qc_rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(qc_rows)


def window_exclusion_summary(
    subject_id: str,
    total_candidate_windows: int,
    kept_windows: int,
) -> dict[str, float | str | int]:
    excluded = total_candidate_windows - kept_windows
    return {
        "subject_id": subject_id,
        "candidate_windows": total_candidate_windows,
        "kept_windows": kept_windows,
        "excluded_windows": excluded,
        "exclusion_rate": excluded / total_candidate_windows if total_candidate_windows else 0.0,
    }
