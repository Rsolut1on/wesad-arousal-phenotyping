"""Label standardization for WESAD protocol states."""

from __future__ import annotations

import numpy as np
import pandas as pd

LABEL_NAMES = {
    1: "baseline",
    2: "stress",
    3: "amusement",
}


def label_config(cfg: dict) -> dict[str, int | list[int]]:
    labels = cfg.get("labels", {})
    return {
        "baseline": int(labels.get("baseline", 1)),
        "stress": int(labels.get("stress", 2)),
        "amusement": int(labels.get("amusement", 3)),
        "exclude": list(labels.get("exclude", [0, 4, 5, 6, 7])),
    }


def valid_label_values(cfg: dict) -> set[int]:
    lc = label_config(cfg)
    return {lc["baseline"], lc["stress"], lc["amusement"]}


def map_label_code(code: int, cfg: dict) -> str | None:
    lc = label_config(cfg)
    mapping = {
        lc["baseline"]: "baseline",
        lc["stress"]: "stress",
        lc["amusement"]: "amusement",
    }
    return mapping.get(int(code))


def to_binary_label(label_name: str) -> int | None:
    if label_name == "stress":
        return 1
    if label_name in {"baseline", "amusement"}:
        return 0
    return None


def standardize_label_array(labels: np.ndarray, cfg: dict) -> np.ndarray:
    lc = label_config(cfg)
    out = labels.astype(int).copy()
    exclude = set(lc["exclude"])
    out[np.isin(out, list(exclude))] = -1
    valid = valid_label_values(cfg)
    out[~np.isin(out, list(valid))] = -1
    return out


def labels_to_names(labels: np.ndarray, cfg: dict) -> np.ndarray:
    names = np.full(labels.shape, None, dtype=object)
    for code in valid_label_values(cfg):
        name = map_label_code(code, cfg)
        names[labels == code] = name
    return names


def summarize_label_coverage(labels: np.ndarray, cfg: dict) -> pd.DataFrame:
    rows = []
    total = len(labels)
    for code in sorted(valid_label_values(cfg)):
        count = int(np.sum(labels == code))
        rows.append(
            {
                "label_code": code,
                "label_name": map_label_code(code, cfg),
                "samples": count,
                "fraction": count / total if total else 0.0,
            }
        )
    excluded = int(np.sum(labels == -1))
    rows.append(
        {
            "label_code": -1,
            "label_name": "excluded",
            "samples": excluded,
            "fraction": excluded / total if total else 0.0,
        }
    )
    return pd.DataFrame(rows)
