"""Model interpretation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from wesad_arousal.modeling.baselines import get_model
from wesad_arousal.modeling.evaluation import _prepare_xy, run_loso_evaluation


def compute_feature_importance(features_df: pd.DataFrame, cfg: dict, model_name: str = "random_forest") -> pd.DataFrame:
    task = cfg.get("modeling", {}).get("task", "binary")
    X, y, groups, feature_cols = _prepare_xy(features_df, task)
    model = get_model(model_name, random_state=cfg.get("modeling", {}).get("random_state", 42))
    model.fit(X, y)

    if hasattr(model[-1], "feature_importances_"):
        values = model[-1].feature_importances_
    else:
        result = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=1)
        values = result.importances_mean

    return pd.DataFrame({"feature": feature_cols, "importance": values}).sort_values(
        "importance", ascending=False
    )


def run_modality_ablation(features_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    groups = {
        "hrv": ["chest_hrv"],
        "eda": ["chest_eda", "wrist_eda"],
        "respiration": ["chest_resp"],
        "bvp": ["wrist_bvp"],
        "motion": ["chest_motion", "wrist_motion"],
        "temperature": ["chest_temp", "wrist_temp"],
        "emg": ["chest_emg"],
    }

    rows = []
    for group_name, prefixes in groups.items():
        cols = [col for col in features_df.columns if any(col.startswith(p) for p in prefixes)]
        if not cols:
            continue
        subset = features_df.copy()
        drop_prefixes = [p for name, ps in groups.items() if name != group_name for p in ps]
        drop_cols = [col for col in subset.columns if any(col.startswith(p) for p in drop_prefixes)]
        subset = subset.drop(columns=drop_cols, errors="ignore")
        results = run_loso_evaluation(subset, cfg, model_names=["random_forest"])
        payload = results["models"]["random_forest"]
        rows.append(
            {
                "ablation": f"without_{group_name}",
                "balanced_accuracy": payload["balanced_accuracy_mean"],
                "macro_f1": payload["macro_f1_mean"],
            }
        )

    full = run_loso_evaluation(features_df, cfg, model_names=["random_forest"])["models"]["random_forest"]
    rows.insert(
        0,
        {
            "ablation": "full",
            "balanced_accuracy": full["balanced_accuracy_mean"],
            "macro_f1": full["macro_f1_mean"],
        },
    )
    return pd.DataFrame(rows)
