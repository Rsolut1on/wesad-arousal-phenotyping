"""Markdown and figure report generation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from wesad_arousal.modeling.interpretation import compute_feature_importance, run_modality_ablation
from wesad_arousal.visualization import (
    plot_ablation,
    plot_confusion_matrix,
    plot_feature_distributions,
    plot_feature_distributions_grid,
    plot_feature_importance,
    plot_metrics_from_json,
    plot_pipeline_diagram,
    plot_qc_heatmap,
    plot_signal_preprocessing_examples,
)


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def render_qc_report(qc_csv: Path, out_path: Path) -> None:
    qc_df = pd.read_csv(qc_csv)
    sample = _df_to_markdown_table(qc_df.head(12))
    body = f"""# Example QC Report

This report summarizes automated quality-control metrics per subject and sensor modality.

## Sample QC Metrics

{sample}

## Notes

- Missingness and flatline fractions flag unreliable segments.
- Window exclusion counts reflect mixed-label windows removed by majority-vote filtering.
- These metrics are intended for transparent screening in decentralized phenotyping workflows.
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")


def generate_full_report(cfg: dict) -> Path:
    reports_dir = Path(cfg["paths"]["reports_dir"])
    figures_dir = reports_dir / "figures"
    tables_dir = reports_dir / "tables"
    outputs_dir = Path(cfg["paths"]["outputs_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])

    plot_pipeline_diagram(figures_dir / "pipeline_diagram.png")
    plot_signal_preprocessing_examples(cfg, figures_dir / "signal_preprocessing_examples.png")

    qc_path = outputs_dir / "qc_metrics.csv"
    if qc_path.exists():
        plot_qc_heatmap(pd.read_csv(qc_path), figures_dir / "qc_heatmap.png")
        render_qc_report(qc_path, reports_dir / "example_qc_report.md")

    features_path = processed_dir / "features.parquet"
    if features_path.exists():
        features_df = pd.read_parquet(features_path)
        plot_feature_distributions(features_df, figures_dir / "feature_distributions.png")
        plot_feature_distributions_grid(features_df, figures_dir / "feature_distributions_grid.png")

    default_metrics = outputs_dir / "metrics_default.json"
    if default_metrics.exists():
        plot_metrics_from_json(default_metrics, figures_dir)

    metrics_files = sorted(outputs_dir.glob("metrics_*.json"))
    if metrics_files:
        with metrics_files[-1].open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        best_model = max(
            metrics["models"],
            key=lambda name: metrics["models"][name]["balanced_accuracy_mean"],
        )
        payload = metrics["models"][best_model]
        labels = ["0", "1"] if metrics["task"] == "binary" else ["baseline", "stress", "amusement"]
        plot_confusion_matrix(payload["confusion_matrix"], labels, figures_dir / "confusion_matrix.png")

        if features_path.exists():
            importance = compute_feature_importance(pd.read_parquet(features_path), cfg, model_name=best_model)
            importance.to_csv(tables_dir / "feature_importance.csv", index=False)
            plot_feature_importance(importance, figures_dir / "feature_importance.png")

            ablation = run_modality_ablation(pd.read_parquet(features_path), cfg)
            ablation.to_csv(tables_dir / "modality_ablation.csv", index=False)
            plot_ablation(ablation, figures_dir / "modality_ablation.png")

    summary_rows = []
    for path in sorted(outputs_dir.glob("metrics_summary_*.csv")):
        df = pd.read_csv(path)
        df["config_tag"] = path.stem.replace("metrics_summary_", "")
        summary_rows.append(df)
    if summary_rows:
        pd.concat(summary_rows, ignore_index=True).to_csv(tables_dir / "results_summary.csv", index=False)

    return reports_dir / "example_qc_report.md"
