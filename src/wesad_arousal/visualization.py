"""Figure generation for reports and README."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from wesad_arousal.data import load_subject_interim
from wesad_arousal.labels import label_config
from wesad_arousal.preprocessing import bandpass_filter, lowpass_filter


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _signal_values(frame: pd.DataFrame) -> np.ndarray:
    if "value" in frame.columns:
        return frame["value"].to_numpy(dtype=float)
    return frame.iloc[:, 0].to_numpy(dtype=float)


def _find_stress_segment(labels: np.ndarray, cfg: dict, duration_s: float = 30.0) -> tuple[int, int]:
    lc = label_config(cfg)
    stress_code = lc["stress"]
    fs = cfg["sampling_rates"]["label"]
    n = int(duration_s * fs)
    idx = np.where(labels == stress_code)[0]
    if idx.size < n:
        start = max(0, len(labels) - n)
        return start, min(len(labels), start + n)
    start = int(idx[len(idx) // 2]) - n // 2
    start = max(0, start)
    return start, min(len(labels), start + n)


def plot_signal_preprocessing_examples(cfg: dict, out_path: Path, subject_id: str = "S2") -> None:
    interim_dir = Path(cfg["paths"]["interim_dir"])
    subject_dir = interim_dir / subject_id
    if not subject_dir.exists():
        subjects = sorted([p.name for p in interim_dir.iterdir() if p.is_dir() and p.name.startswith("S")])
        if not subjects:
            return
        subject_id = subjects[0]
        subject_dir = interim_dir / subject_id

    frames = load_subject_interim(interim_dir, subject_id)
    labels = np.asarray(frames["labels"])
    start, end = _find_stress_segment(labels, cfg)
    fs_chest = cfg["sampling_rates"]["chest"]
    t = np.arange(end - start) / fs_chest

    ecg = _signal_values(frames["chest_ECG"])[start:end]
    eda = _signal_values(frames["chest_EDA"])[start:end]
    resp = _signal_values(frames["chest_Resp"])[start:end]
    ecg_f = bandpass_filter(ecg, fs_chest, 0.5, 40.0)
    eda_f = lowpass_filter(eda, fs_chest, 1.0)
    resp_f = lowpass_filter(resp, fs_chest, 0.5)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, ecg, color="#B0BEC5", linewidth=0.8, label="raw")
    axes[0].plot(t, ecg_f, color="#E45756", linewidth=1.0, label="bandpass 0.5–40 Hz")
    axes[0].set_ylabel("ECG")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title(f"Wearable Signal Preprocessing ({subject_id}, stress segment)")

    axes[1].plot(t, eda, color="#B0BEC5", linewidth=0.8, label="raw")
    axes[1].plot(t, eda_f, color="#4C78A8", linewidth=1.0, label="lowpass 1 Hz")
    axes[1].set_ylabel("EDA")
    axes[1].legend(loc="upper right", fontsize=8)

    axes[2].plot(t, resp, color="#B0BEC5", linewidth=0.8, label="raw")
    axes[2].plot(t, resp_f, color="#72B7B2", linewidth=1.0, label="lowpass 0.5 Hz")
    axes[2].set_ylabel("Respiration")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right", fontsize=8)

    _save(fig, out_path)


def plot_pipeline_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    stages = [
        "WESAD .pkl",
        "Preprocess",
        "QC",
        "Windowing",
        "Features",
        "LOSO Models",
        "Reports",
    ]
    xs = np.linspace(0.05, 0.95, len(stages))
    for x, stage in zip(xs, stages):
        ax.text(x, 0.5, stage, ha="center", va="center", bbox=dict(boxstyle="round", fc="#E8F1FF"))
    for i in range(len(xs) - 1):
        ax.annotate("", xy=(xs[i + 1] - 0.05, 0.5), xytext=(xs[i] + 0.05, 0.5), arrowprops=dict(arrowstyle="->"))
    ax.set_title("WESAD Arousal Phenotyping Pipeline")
    _save(fig, out_path)


def plot_qc_heatmap(qc_df: pd.DataFrame, out_path: Path) -> None:
    numeric = qc_df.select_dtypes(include=[np.number])
    if numeric.empty:
        return
    pivot = qc_df.pivot_table(
        index="subject_id",
        columns="signal",
        values="missingness",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax)
    ax.set_title("Signal Missingness by Subject and Modality")
    _save(fig, out_path)


def plot_feature_distributions(features_df: pd.DataFrame, out_path: Path) -> None:
    candidates = [
        col
        for col in features_df.columns
        if col.startswith(("chest_hrv_hr_mean", "chest_eda_mean", "wrist_eda_mean", "chest_resp_rate_mean"))
    ]
    if not candidates:
        candidates = [col for col in features_df.columns if col.startswith(("chest_eda", "chest_hrv", "wrist_eda"))]
    if not candidates:
        return
    col = candidates[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=features_df, x="label_name", y=col, hue="label_name", ax=ax, palette="Set2", legend=False)
    ax.set_title(f"Feature Distribution by Condition: {col}")
    ax.set_xlabel("Condition")
    _save(fig, out_path)


def plot_feature_distributions_grid(features_df: pd.DataFrame, out_path: Path) -> None:
    candidates = [
        "chest_hrv_hr_mean",
        "chest_eda_phasic_mean",
        "chest_resp_rate_mean",
        "wrist_bvp_pulse_rate",
        "wrist_motion_vm_mean",
        "chest_temp_mean",
    ]
    cols = [c for c in candidates if c in features_df.columns]
    if not cols:
        return

    n = len(cols)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    axes_flat = axes.flatten()
    for ax, col in zip(axes_flat, cols):
        sns.boxplot(data=features_df, x="label_name", y=col, hue="label_name", ax=ax, palette="Set2", legend=False)
        ax.set_title(col.replace("_", " "), fontsize=9)
        ax.tick_params(axis="x", rotation=20)
    for ax in axes_flat[len(cols):]:
        ax.axis("off")
    fig.suptitle("Feature Distributions by Condition (LOSO windows)", y=1.02)
    _save(fig, out_path)


def plot_confusion_matrix(cm: list[list[int]], labels: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(np.asarray(cm), annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("LOSO Confusion Matrix")
    _save(fig, out_path)


def plot_ablation(ablation_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=ablation_df, x="ablation", y="balanced_accuracy", ax=ax, color="#4C78A8")
    ax.tick_params(axis="x", rotation=30)
    ax.set_title("Modality Ablation (Random Forest, LOSO)")
    _save(fig, out_path)


def plot_feature_importance(importance_df: pd.DataFrame, out_path: Path, top_n: int = 15) -> None:
    top = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=top, y="feature", x="importance", ax=ax, color="#72B7B2")
    ax.set_title("Top Feature Importances (LOSO-trained model)")
    _save(fig, out_path)


def plot_loso_per_subject(metrics: dict, model_name: str, out_path: Path) -> None:
    payload = metrics.get("models", {}).get(model_name)
    if not payload or "per_subject" not in payload:
        return
    df = pd.DataFrame(payload["per_subject"])
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=df, x="held_out_subject", y="balanced_accuracy", ax=ax, color="#4C78A8")
    mean_acc = payload["balanced_accuracy_mean"]
    ax.axhline(mean_acc, color="#E45756", linestyle="--", linewidth=1.2, label=f"mean = {mean_acc:.2f}")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Subject-Independent Evaluation (LOSO) — {model_name}")
    ax.set_xlabel("Held-out subject")
    ax.set_ylabel("Balanced accuracy")
    ax.legend()
    _save(fig, out_path)


def plot_model_comparison(metrics: dict, out_path: Path) -> None:
    rows = []
    for name, payload in metrics.get("models", {}).items():
        rows.append(
            {
                "model": name,
                "balanced_accuracy": payload["balanced_accuracy_mean"],
                "macro_f1": payload["macro_f1_mean"],
            }
        )
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=df, x="model", y="balanced_accuracy", ax=ax, color="#4C78A8")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"LOSO Model Comparison ({metrics.get('task', 'task')})")
    ax.set_xlabel("Model")
    ax.set_ylabel("Balanced accuracy")
    _save(fig, out_path)


def plot_metrics_from_json(metrics_path: Path, figures_dir: Path) -> None:
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    best_model = max(
        metrics["models"],
        key=lambda name: metrics["models"][name]["balanced_accuracy_mean"],
    )
    plot_model_comparison(metrics, figures_dir / "loso_model_comparison.png")
    plot_loso_per_subject(metrics, best_model, figures_dir / "loso_per_subject.png")
