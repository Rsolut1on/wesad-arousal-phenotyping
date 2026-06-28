"""WESAD data ingestion and interim storage."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from wesad_arousal.labels import standardize_label_array


CHEST_KEYS = ["ACC", "ECG", "EMG", "EDA", "Temp", "Resp"]
WRIST_KEYS = ["ACC", "BVP", "EDA", "TEMP"]


def discover_subjects(raw_dir: Path, subject_ids: list[int] | None = None) -> list[str]:
    if subject_ids is not None:
        return [f"S{sid}" for sid in subject_ids]

    subjects = []
    for path in sorted(raw_dir.iterdir()):
        if path.is_dir() and re.match(r"^S\d+$", path.name):
            pkl = path / f"{path.name}.pkl"
            if pkl.exists():
                subjects.append(path.name)
    return subjects


def load_subject_pkl(pkl_path: Path) -> dict[str, Any]:
    with pkl_path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _flatten_signal(signal: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal)
    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0].astype(float)
    return arr.astype(float)


def _acc_columns(signal: np.ndarray) -> pd.DataFrame:
    arr = np.asarray(signal)
    if arr.ndim == 2 and arr.shape[1] == 3:
        return pd.DataFrame(arr, columns=["x", "y", "z"])
    flat = _flatten_signal(signal)
    return pd.DataFrame({"value": flat})


def subject_record_to_frames(subject_id: str, record: dict[str, Any], cfg: dict) -> dict[str, pd.DataFrame | np.ndarray]:
    labels = standardize_label_array(np.asarray(record["label"]).reshape(-1), cfg)
    fs_label = cfg["sampling_rates"]["label"]

    frames: dict[str, pd.DataFrame | np.ndarray] = {"labels": labels}

    chest = record["signal"]["chest"]
    for key in CHEST_KEYS:
        if key not in chest:
            continue
        if key == "ACC":
            df = _acc_columns(chest[key])
        else:
            df = pd.DataFrame({"value": _flatten_signal(chest[key])})
        df["device"] = "chest"
        df["signal"] = key
        df["subject_id"] = subject_id
        df["sample_index"] = np.arange(len(df))
        df["timestamp_s"] = df["sample_index"] / cfg["sampling_rates"]["chest"]
        frames[f"chest_{key}"] = df

    wrist = record["signal"]["wrist"]
    wrist_fs = cfg["sampling_rates"]["wrist"]
    for key in WRIST_KEYS:
        if key not in wrist:
            continue
        if key == "ACC":
            df = _acc_columns(wrist[key])
        else:
            df = pd.DataFrame({"value": _flatten_signal(wrist[key])})
        df["device"] = "wrist"
        df["signal"] = key
        df["subject_id"] = subject_id
        df["sample_index"] = np.arange(len(df))
        df["timestamp_s"] = df["sample_index"] / wrist_fs[key]
        frames[f"wrist_{key}"] = df

    label_df = pd.DataFrame(
        {
            "subject_id": subject_id,
            "sample_index": np.arange(len(labels)),
            "timestamp_s": np.arange(len(labels)) / fs_label,
            "label_code": labels,
        }
    )
    frames["label_timeline"] = label_df
    return frames


def save_subject_interim(frames: dict[str, pd.DataFrame | np.ndarray], out_dir: Path) -> None:
    subject_id = None
    for value in frames.values():
        if isinstance(value, pd.DataFrame) and "subject_id" in value.columns:
            subject_id = value["subject_id"].iloc[0]
            break
    if subject_id is None:
        raise ValueError("Could not infer subject_id from frames")

    subject_dir = out_dir / subject_id
    subject_dir.mkdir(parents=True, exist_ok=True)

    for name, payload in frames.items():
        if isinstance(payload, pd.DataFrame):
            payload.to_parquet(subject_dir / f"{name}.parquet", index=False)
        else:
            np.save(subject_dir / f"{name}.npy", payload)


def load_subject_interim(interim_dir: Path, subject_id: str) -> dict[str, pd.DataFrame | np.ndarray]:
    subject_dir = interim_dir / subject_id
    if not subject_dir.exists():
        raise FileNotFoundError(f"Interim data missing for {subject_id}")

    frames: dict[str, pd.DataFrame | np.ndarray] = {}
    for path in sorted(subject_dir.glob("*")):
        if path.suffix == ".parquet":
            frames[path.stem] = pd.read_parquet(path)
        elif path.suffix == ".npy":
            frames[path.stem] = np.load(path)
    return frames


def preprocess_all_subjects(cfg: dict, raw_dir: Path | None = None, out_dir: Path | None = None) -> list[str]:
    raw_dir = Path(raw_dir or cfg["paths"]["raw_dir"])
    out_dir = Path(out_dir or cfg["paths"]["interim_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_ids = cfg.get("subjects", {}).get("ids")
    subjects = discover_subjects(raw_dir, subject_ids)
    if not subjects:
        raise FileNotFoundError(
            f"No WESAD subject folders found under {raw_dir}. "
            "See scripts/download_instructions.md"
        )

    processed = []
    for subject_id in tqdm(subjects, desc="Preprocessing subjects"):
        pkl_path = raw_dir / subject_id / f"{subject_id}.pkl"
        record = load_subject_pkl(pkl_path)
        frames = subject_record_to_frames(subject_id, record, cfg)
        save_subject_interim(frames, out_dir)
        processed.append(subject_id)
    return processed
