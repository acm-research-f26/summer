"""Reporting helpers for extracted fibroid feature tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from fibroid_cavity.constants import GROUP_COLUMN, PREDICTOR_COLUMNS, TARGET_COLUMN
from fibroid_cavity.plotting import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt


def make_feature_report(
    features: pd.DataFrame,
    output_dir: Path,
    predictors: Optional[list[str]] = None,
    target_column: str = TARGET_COLUMN,
    group_column: str = GROUP_COLUMN,
) -> None:
    """Create tabular and visual summaries for extracted fibroid features."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictors = predictors or PREDICTOR_COLUMNS
    available_predictors = [column for column in predictors if column in features.columns]

    _write_dataset_summary(features, output_dir, target_column, group_column)
    _write_missingness(features, output_dir)
    _write_correlations(features, output_dir, available_predictors)
    _plot_class_balance(features, output_dir, target_column)
    _plot_feature_distributions(features, output_dir, available_predictors, target_column)
    _plot_volume_distance(features, output_dir, target_column)


def _write_dataset_summary(
    features: pd.DataFrame,
    output_dir: Path,
    target_column: str,
    group_column: str,
) -> None:
    summary = {
        "n_fibroids": int(len(features)),
        "n_patients": int(features[group_column].nunique()) if group_column in features.columns else None,
        "columns": list(features.columns),
    }

    if target_column in features.columns:
        counts = features[target_column].value_counts(dropna=False).sort_index()
        summary["target_counts"] = {str(key): int(value) for key, value in counts.items()}

    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _write_missingness(features: pd.DataFrame, output_dir: Path) -> None:
    missingness = pd.DataFrame(
        {
            "column": features.columns,
            "missing_count": features.isna().sum().to_numpy(),
            "missing_ratio": features.isna().mean().to_numpy(),
        }
    ).sort_values(["missing_ratio", "column"], ascending=[False, True])
    missingness.to_csv(output_dir / "feature_missingness.csv", index=False)


def _write_correlations(features: pd.DataFrame, output_dir: Path, predictors: list[str]) -> None:
    if len(predictors) < 2:
        return
    correlations = features[predictors].corr(numeric_only=True)
    correlations.to_csv(output_dir / "feature_correlations.csv")


def _plot_class_balance(features: pd.DataFrame, output_dir: Path, target_column: str) -> None:
    if target_column not in features.columns:
        return

    counts = features[target_column].value_counts().sort_index()
    labels = ["not touching" if value == 0 else "touching" for value in counts.index]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, counts.to_numpy(), color=["#4e79a7", "#e15759"][: len(counts)])
    plt.ylabel("Fibroid count")
    plt.title("Cavity-Contact Class Balance")
    plt.tight_layout()
    plt.savefig(output_dir / "class_balance.png", dpi=200)
    plt.close()


def _plot_feature_distributions(
    features: pd.DataFrame,
    output_dir: Path,
    predictors: list[str],
    target_column: str,
) -> None:
    if not predictors:
        return

    n_features = len(predictors)
    n_cols = min(2, n_features)
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feature in zip(axes, predictors):
        if target_column in features.columns:
            for target_value, rows in features.groupby(target_column):
                label = "touching" if int(target_value) == 1 else "not touching"
                ax.hist(rows[feature].dropna(), bins=15, alpha=0.6, label=label)
            ax.legend()
        else:
            ax.hist(features[feature].dropna(), bins=15, alpha=0.8)

        ax.set_title(feature)
        ax.set_ylabel("Count")

    for ax in axes[len(predictors) :]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_dir / "feature_distributions.png", dpi=200)
    plt.close(fig)


def _plot_volume_distance(features: pd.DataFrame, output_dir: Path, target_column: str) -> None:
    required = {"volume_mm3", "centroid_to_cavity_dist_mm"}
    if not required.issubset(features.columns):
        return

    plt.figure(figsize=(6, 5))
    if target_column in features.columns:
        colors = features[target_column].map({0: "#4e79a7", 1: "#e15759"}).fillna("#767676")
    else:
        colors = "#4e79a7"

    plt.scatter(
        features["centroid_to_cavity_dist_mm"],
        features["volume_mm3"],
        c=colors,
        alpha=0.75,
        edgecolor="white",
        linewidth=0.5,
    )
    plt.xlabel("Centroid to cavity distance (mm)")
    plt.ylabel("Volume (mm3)")
    plt.title("Fibroid Volume vs. Cavity Distance")
    plt.tight_layout()
    plt.savefig(output_dir / "volume_vs_distance.png", dpi=200)
    plt.close()
