"""Helpers for loading and visualizing Vertex AI training artifacts from GCS."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

try:
    from gcs import (
        GCS_BUCKET,
        download_artifact_prefix,
        get_latest_run_prefix,
    )
except ImportError:
    from src.gcs import (
        GCS_BUCKET,
        download_artifact_prefix,
        get_latest_run_prefix,
    )


def load_run_artifacts(
    run_prefix: str | None = None,
    local_dir: str | Path = "artifacts",
    bucket: str = GCS_BUCKET,
    project_id: str | None = None,
) -> Path:
    """Download a training run's artifacts from GCS to a local directory."""
    prefix = run_prefix or get_latest_run_prefix(bucket=bucket, project_id=project_id)
    if prefix is None:
        raise FileNotFoundError("No training runs found under output/models/")
    return download_artifact_prefix(
        prefix, local_dir, bucket=bucket, project_id=project_id
    )


def load_metrics(artifacts_dir: str | Path) -> dict:
    """Load metrics.json from a downloaded artifacts directory."""
    metrics_path = Path(artifacts_dir) / "metrics.json"
    return json.loads(metrics_path.read_text())


def display_training_summary(artifacts_dir: str | Path) -> dict:
    """Print a concise summary of training metrics."""
    metrics = load_metrics(artifacts_dir)
    print("Hyperparameters:", metrics.get("hyperparameters"))
    print("Class distribution:", metrics.get("class_distribution"))
    print("Final train loss:", metrics.get("final_train_loss"))
    print("Final val loss:", metrics.get("final_val_loss"))
    return metrics


def plot_loss_curve_from_artifacts(artifacts_dir: str | Path) -> Figure:
    """Display the saved loss curve image or rebuild from metrics.csv."""
    loss_path = Path(artifacts_dir) / "loss_curve.png"
    if loss_path.exists():
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(plt.imread(loss_path))
        ax.axis("off")
        ax.set_title("Loss Curve")
        return fig

    import pandas as pd

    history = pd.read_csv(Path(artifacts_dir) / "metrics.csv")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["epoch"], history["train_loss"], marker="o", label="Train Loss")
    ax.plot(history["epoch"], history["val_loss"], marker="o", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.set_title("Training and Validation Loss")
    fig.tight_layout()
    return fig


def plot_confusion_matrix_from_artifacts(artifacts_dir: str | Path) -> Figure:
    """Display the saved confusion matrix image."""
    matrix_path = Path(artifacts_dir) / "confusion_matrix.png"
    if not matrix_path.exists():
        raise FileNotFoundError(f"Missing confusion matrix at {matrix_path}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(plt.imread(matrix_path))
    ax.axis("off")
    ax.set_title("Confusion Matrix")
    return fig
