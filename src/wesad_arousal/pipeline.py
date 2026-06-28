"""End-to-end feature extraction from interim subject data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from wesad_arousal.data import discover_subjects, load_subject_interim
from wesad_arousal.features.eda import extract_eda_features
from wesad_arousal.features.hrv import extract_hrv_features
from wesad_arousal.features.motion import extract_emg_features, extract_motion_features
from wesad_arousal.features.respiration import extract_bvp_features, extract_respiration_features
from wesad_arousal.features.temperature import extract_temperature_features
from wesad_arousal.features.windowing import generate_windows, slice_signal_by_window, windows_to_dataframe
from wesad_arousal.quality import compute_signal_qc, qc_row, window_exclusion_summary
from wesad_arousal.labels import summarize_label_coverage


def _get_signal_array(frame: pd.DataFrame) -> np.ndarray:
    if {"x", "y", "z"}.issubset(frame.columns):
        return frame[["x", "y", "z"]].to_numpy(dtype=float)
    return frame["value"].to_numpy(dtype=float)


def _count_candidate_windows(n_samples: int, fs: float, cfg: dict) -> int:
    window_samples = int(cfg["windowing"]["window_seconds"] * fs)
    step_samples = int(cfg["windowing"]["step_seconds"] * fs)
    if n_samples < window_samples or step_samples <= 0:
        return 0
    return 1 + (n_samples - window_samples) // step_samples


def extract_subject_features(subject_id: str, cfg: dict, interim_dir: Path) -> tuple[pd.DataFrame, list[dict], dict]:
    frames = load_subject_interim(interim_dir, subject_id)
    labels = frames["labels"]
    fs_label = cfg["sampling_rates"]["label"]

    windows = generate_windows(
        labels=np.asarray(labels),
        fs=fs_label,
        window_seconds=cfg["windowing"]["window_seconds"],
        step_seconds=cfg["windowing"]["step_seconds"],
        cfg=cfg,
    )
    candidate = _count_candidate_windows(len(labels), fs_label, cfg)
    exclusion = window_exclusion_summary(subject_id, candidate, len(windows))

    use_chest = cfg["devices"]["use_chest"]
    use_wrist = cfg["devices"]["use_wrist"]
    groups = set(cfg.get("feature_groups", []))

    feature_rows = []
    qc_rows: list[dict] = []

    for window in windows:
        row = {
            "subject_id": subject_id,
            "window_start_s": window.start_s,
            "window_end_s": window.end_s,
            "label_code": window.label_code,
            "label_name": window.label_name,
            "binary_label": window.binary_label,
            "label_purity": window.label_purity,
        }

        if use_chest and "hrv" in groups and "chest_ECG" in frames:
            ecg = _get_signal_array(frames["chest_ECG"])
            segment = slice_signal_by_window(ecg, cfg["sampling_rates"]["chest"], window)
            row.update(extract_hrv_features(segment, cfg["sampling_rates"]["chest"], prefix="chest_hrv"))

        if use_chest and "eda" in groups and "chest_EDA" in frames:
            eda = _get_signal_array(frames["chest_EDA"])
            segment = slice_signal_by_window(eda, cfg["sampling_rates"]["chest"], window)
            row.update(extract_eda_features(segment, cfg["sampling_rates"]["chest"], prefix="chest_eda"))

        if use_wrist and "eda" in groups and "wrist_EDA" in frames:
            eda = _get_signal_array(frames["wrist_EDA"])
            segment = slice_signal_by_window(eda, cfg["sampling_rates"]["wrist"]["EDA"], window)
            row.update(extract_eda_features(segment, cfg["sampling_rates"]["wrist"]["EDA"], prefix="wrist_eda"))

        if use_chest and "respiration" in groups and "chest_Resp" in frames:
            resp = _get_signal_array(frames["chest_Resp"])
            segment = slice_signal_by_window(resp, cfg["sampling_rates"]["chest"], window)
            row.update(extract_respiration_features(segment, cfg["sampling_rates"]["chest"], prefix="chest_resp"))

        if use_wrist and "bvp" in groups and "wrist_BVP" in frames:
            bvp = _get_signal_array(frames["wrist_BVP"])
            segment = slice_signal_by_window(bvp, cfg["sampling_rates"]["wrist"]["BVP"], window)
            row.update(extract_bvp_features(segment, cfg["sampling_rates"]["wrist"]["BVP"], prefix="wrist_bvp"))

        if use_chest and "motion" in groups and "chest_ACC" in frames:
            acc = _get_signal_array(frames["chest_ACC"])
            segment = slice_signal_by_window(acc, cfg["sampling_rates"]["chest"], window)
            row.update(extract_motion_features(segment, cfg["sampling_rates"]["chest"], prefix="chest_motion"))

        if use_wrist and "motion" in groups and "wrist_ACC" in frames:
            acc = _get_signal_array(frames["wrist_ACC"])
            segment = slice_signal_by_window(acc, cfg["sampling_rates"]["wrist"]["ACC"], window)
            row.update(extract_motion_features(segment, cfg["sampling_rates"]["wrist"]["ACC"], prefix="wrist_motion"))

        if use_chest and "temperature" in groups and "chest_Temp" in frames:
            temp = _get_signal_array(frames["chest_Temp"])
            segment = slice_signal_by_window(temp, cfg["sampling_rates"]["chest"], window)
            row.update(extract_temperature_features(segment, cfg["sampling_rates"]["chest"], prefix="chest_temp"))

        if use_wrist and "temperature" in groups and "wrist_TEMP" in frames:
            temp = _get_signal_array(frames["wrist_TEMP"])
            segment = slice_signal_by_window(temp, cfg["sampling_rates"]["wrist"]["TEMP"], window)
            row.update(extract_temperature_features(segment, cfg["sampling_rates"]["wrist"]["TEMP"], prefix="wrist_temp"))

        if use_chest and "emg" in groups and "chest_EMG" in frames:
            emg = _get_signal_array(frames["chest_EMG"])
            segment = slice_signal_by_window(emg, cfg["sampling_rates"]["chest"], window)
            row.update(extract_emg_features(segment, cfg["sampling_rates"]["chest"], prefix="chest_emg"))

        feature_rows.append(row)

    for key, frame in frames.items():
        if not key.startswith(("chest_", "wrist_")):
            continue
        device, signal_name = key.split("_", 1)
        arr = _get_signal_array(frame)
        if signal_name == "ACC":
            fs = cfg["sampling_rates"]["chest"] if device == "chest" else cfg["sampling_rates"]["wrist"]["ACC"]
        elif device == "chest":
            fs = cfg["sampling_rates"]["chest"]
        else:
            fs = cfg["sampling_rates"]["wrist"].get(signal_name, 4)
        metrics = compute_signal_qc(arr, signal_name, device, fs)
        qc_rows.append(qc_row(subject_id, device, signal_name, metrics))

    qc_rows.append(
        {
            "subject_id": subject_id,
            "device": "all",
            "signal": "window_exclusion",
            **exclusion,
        }
    )

    features_df = pd.DataFrame(feature_rows)
    label_summary = summarize_label_coverage(np.asarray(labels), cfg)
    return features_df, qc_rows, {"windows": windows_to_dataframe(windows, subject_id), "labels": label_summary}


def extract_all_features(cfg: dict, interim_dir: Path | None = None) -> pd.DataFrame:
    interim_dir = Path(interim_dir or cfg["paths"]["interim_dir"])
    raw_dir = Path(cfg["paths"]["raw_dir"])
    subjects = discover_subjects(raw_dir, cfg.get("subjects", {}).get("ids"))

    all_features = []
    all_qc = []
    for subject_id in tqdm(subjects, desc="Extracting features"):
        subject_dir = interim_dir / subject_id
        if not subject_dir.exists():
            continue
        features, qc_rows, _ = extract_subject_features(subject_id, cfg, interim_dir)
        all_features.append(features)
        all_qc.extend(qc_rows)

    if not all_features:
        raise FileNotFoundError("No interim subject data found. Run preprocessing first.")

    features_df = pd.concat(all_features, ignore_index=True)
    qc_df = pd.DataFrame(all_qc)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(processed_dir / "features.parquet", index=False)
    features_df.to_csv(processed_dir / "features.csv", index=False)

    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    qc_df.to_csv(outputs_dir / "qc_metrics.csv", index=False)

    return features_df
