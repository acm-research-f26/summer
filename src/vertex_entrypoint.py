"""Training entrypoint for the data center vision classifier (local + Vertex AI)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from .config import Config, load_config
    from .logging_config import log_epoch_metrics, setup_logging
    from .model_def import DataCenterVisionNet
except ImportError:
    try:
        from config import Config, load_config
        from logging_config import log_epoch_metrics, setup_logging
        from model_def import DataCenterVisionNet
    except ImportError:
        from src.config import Config, load_config
        from src.logging_config import log_epoch_metrics, setup_logging
        from src.model_def import DataCenterVisionNet

logger = setup_logging()


def resolve_device(device: str | None = None, config: Config | None = None) -> torch.device:
    """Resolve the compute device for training.

    Priority for "auto": MPS (Apple Silicon) > CUDA > CPU.
    """
    cfg = _cfg_lazy(config)
    spec = device if device is not None else cfg.training.device
    if spec == "auto":
        if torch.backends.mps.is_available():
            chosen = torch.device("mps")
        elif torch.cuda.is_available():
            chosen = torch.device("cuda")
        else:
            chosen = torch.device("cpu")
    else:
        chosen = torch.device(spec)
    logger.info("Training device: %s", chosen)
    return chosen


def _cfg_lazy(config: Config | None) -> Config:
    return config or load_config()


def _cfg(config: Config | None) -> Config:
    return config or load_config()


class VisionImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset loading .npy tiles and labels from the training manifest."""

    def __init__(self, csv_file: str | Path, image_dir: str | Path) -> None:
        self.manifest = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.manifest.iloc[index]
        img_path = self.image_dir / f"tile_{int(row['OBJECTID'])}.npy"
        image = np.load(img_path)
        label = int(row["target_label"])
        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )


def compute_class_weights(
    manifest: pd.DataFrame,
    num_classes: int | None = None,
    config: Config | None = None,
) -> torch.Tensor:
    """Compute inverse-frequency class weights normalized to num_classes."""
    cfg = _cfg(config)
    if num_classes is not None:
        resolved_num_classes = num_classes
    else:
        resolved_num_classes = cfg.model.num_classes
    counts = (
        manifest["target_label"]
        .value_counts()
        .reindex(range(resolved_num_classes), fill_value=1)
    )
    weights = 1.0 / counts.astype(float)
    normalized = weights / weights.sum() * resolved_num_classes
    return torch.tensor(normalized.values, dtype=torch.float32)


def stratified_train_val_indices(
    manifest: pd.DataFrame,
    val_fraction: float | None = None,
    random_state: int | None = None,
    config: Config | None = None,
) -> tuple[list[int], list[int]]:
    """Split manifest indices into stratified train and validation sets."""
    cfg = _cfg(config)
    resolved_val_fraction = (
        val_fraction if val_fraction is not None else cfg.training.val_fraction
    )
    resolved_random_state = (
        random_state if random_state is not None else cfg.training.random_state
    )
    rng = np.random.default_rng(resolved_random_state)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for label in sorted(manifest["target_label"].unique()):
        label_rows = manifest[manifest["target_label"] == label]
        indices = label_rows.index.to_numpy(copy=True)
        rng.shuffle(indices)
        n_val = int(len(indices) * resolved_val_fraction)
        n_val = max(1, n_val) if len(indices) > 1 else 0
        val_indices.extend(indices[:n_val].tolist())
        train_indices.extend(indices[n_val:].tolist())

    return train_indices, val_indices


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device | None = None,
) -> float:
    resolved_device = device or torch.device("cpu")
    is_train = optimizer is not None
    model.train(is_train)
    epoch_loss = 0.0
    batch_count = 0

    for imgs, labels in dataloader:
        imgs = imgs.to(resolved_device)
        labels = labels.to(resolved_device)
        if is_train:
            optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        if is_train:
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()
        batch_count += 1

    return epoch_loss / max(batch_count, 1)


def _collect_predictions(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    resolved_device = device or torch.device("cpu")
    model.eval()
    preds: list[int] = []
    labels: list[int] = []
    with torch.no_grad():
        for imgs, batch_labels in dataloader:
            imgs = imgs.to(resolved_device)
            batch_labels = batch_labels.to(resolved_device)
            outputs = model(imgs)
            preds.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            labels.extend(batch_labels.cpu().numpy().tolist())
    return np.array(preds), np.array(labels)


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    num_classes: int,
    plot_dpi: int,
) -> None:
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred, strict=True):
        matrix[int(true_label), int(pred_label)] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Validation Confusion Matrix")
    for row in range(num_classes):
        for col in range(num_classes):
            ax.text(
                col, row, str(matrix[row, col]), ha="center", va="center", color="black"
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=plot_dpi)
    plt.close(fig)


def _save_loss_curve(
    history: list[dict[str, float]],
    output_path: Path,
    plot_dpi: int,
) -> None:
    epochs = [row["epoch"] for row in history]
    train_losses = [row["train_loss"] for row in history]
    val_losses = [row["val_loss"] for row in history]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_losses, marker="o", label="Train Loss")
    ax.plot(epochs, val_losses, marker="o", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=plot_dpi)
    plt.close(fig)


def _compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
    class_labels: dict[int, str] | None = None,
) -> dict:
    """Compute overall accuracy and per-class precision, recall, F1 from arrays."""
    overall_accuracy = float(np.mean(y_true == y_pred))

    per_class: dict[str, dict[str, float]] = {}
    for c in range(num_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        label = class_labels.get(c, str(c)) if class_labels else str(c)
        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(np.sum(y_true == c)),
        }

    # Macro averages (unweighted mean across classes).
    macro_precision = float(np.mean([v["precision"] for v in per_class.values()]))
    macro_recall = float(np.mean([v["recall"] for v in per_class.values()]))
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()]))

    return {
        "overall_accuracy": round(overall_accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }


def _save_training_artifacts(
    model_dir: Path,
    history: list[dict[str, float]],
    manifest: pd.DataFrame,
    hyperparams: dict[str, float | int],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: Config,
) -> None:
    class_distribution = manifest["target_label"].value_counts().sort_index().to_dict()
    class_labels = {int(k): v for k, v in config.eda.impact_tier_labels.items()}
    classification_metrics = _compute_classification_metrics(
        y_true, y_pred, num_classes=config.model.num_classes, class_labels=class_labels
    )
    metrics = {
        "hyperparameters": hyperparams,
        "class_distribution": class_distribution,
        "classification_metrics": classification_metrics,
        "epoch_history": history,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_val_loss": history[-1]["val_loss"] if history else None,
    }
    metrics_path = model_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    plot_dpi = config.training.plot_dpi
    pd.DataFrame(history).to_csv(model_dir / "metrics.csv", index=False)
    _save_loss_curve(history, model_dir / "loss_curve.png", plot_dpi)
    _save_confusion_matrix(
        y_true,
        y_pred,
        model_dir / "confusion_matrix.png",
        num_classes=config.model.num_classes,
        plot_dpi=plot_dpi,
    )

    logger.info("Artifacts saved to %s", model_dir)


def train(
    training_dir: str | Path,
    model_dir: str | Path,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    val_fraction: float | None = None,
    device: str | torch.device | None = None,
    config: Config | None = None,
) -> None:
    """Run the training loop and persist model weights plus eval artifacts."""
    cfg = _cfg(config)
    train_cfg = cfg.training
    gcs_cfg = cfg.gcs

    resolved_epochs = epochs if epochs is not None else train_cfg.epochs
    resolved_batch_size = batch_size if batch_size is not None else train_cfg.batch_size
    resolved_learning_rate = (
        learning_rate if learning_rate is not None else train_cfg.learning_rate
    )
    resolved_val_fraction = (
        val_fraction if val_fraction is not None else train_cfg.val_fraction
    )

    if isinstance(device, torch.device):
        resolved_device = device
    else:
        resolved_device = resolve_device(device, cfg)

    training_path = Path(training_dir)
    csv_path = training_path / gcs_cfg.manifest_blob
    image_path = training_path / gcs_cfg.prefixes.image_tiles

    manifest = pd.read_csv(csv_path)
    class_counts = manifest["target_label"].value_counts().sort_index()
    logger.info("Class distribution: %s", class_counts.to_dict())

    full_dataset = VisionImageDataset(csv_path, image_path)
    train_idx, val_idx = stratified_train_val_indices(
        manifest,
        val_fraction=resolved_val_fraction,
        config=cfg,
    )
    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=resolved_batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=resolved_batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = DataCenterVisionNet(config=cfg).to(resolved_device)
    class_weights = compute_class_weights(manifest, config=cfg).to(resolved_device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=resolved_learning_rate)

    history: list[dict[str, float]] = []
    logger.info(
        "Beginning training loop (%d train / %d val samples) on %s...",
        len(train_idx),
        len(val_idx),
        resolved_device,
    )
    for epoch in range(resolved_epochs):
        try:
            train_loss = _run_epoch(model, train_loader, criterion, optimizer, resolved_device)
            val_loss = _run_epoch(model, val_loader, criterion, optimizer=None, device=resolved_device)
            current_lr = optimizer.param_groups[0]["lr"]
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr,
                }
            )
            log_epoch_metrics(
                logger,
                epoch=epoch + 1,
                total_epochs=resolved_epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
            )
        except Exception:
            logger.exception("Training failed during epoch %d", epoch + 1)
            raise

    model_output_path = Path(model_dir)
    model_output_path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_output_path / "model.pth")
    logger.info("Model weights saved to %s", model_output_path / "model.pth")

    y_pred, y_true = _collect_predictions(model, val_loader, resolved_device)
    _save_training_artifacts(
        model_output_path,
        history,
        manifest,
        hyperparams={
            "epochs": resolved_epochs,
            "batch_size": resolved_batch_size,
            "learning_rate": resolved_learning_rate,
            "val_fraction": resolved_val_fraction,
            "device": str(resolved_device),
        },
        y_true=y_true,
        y_pred=y_pred,
        config=cfg,
    )


def _gs_uri_to_fuse_path(uri: str) -> Path:
    """Map gs://bucket/key to the Vertex AI GCS FUSE mount at /gcs/bucket/key."""
    if not uri.startswith("gs://"):
        return Path(uri)
    without_scheme = uri.removeprefix("gs://")
    bucket, _, key = without_scheme.partition("/")
    return Path("/gcs") / bucket / key


def resolve_training_dir(training_arg: str, config: Config | None = None) -> Path:
    """Resolve training directory from CLI arg, GCS fuse mount, or local fallback."""
    cfg = _cfg(config)
    candidate = Path(training_arg)
    if candidate.exists():
        return candidate
    fuse_path = Path(cfg.gcs.fuse_root)
    if fuse_path.exists():
        return fuse_path
    return candidate


def resolve_model_dir(model_arg: str, config: Config | None = None) -> Path:
    """Resolve model output directory, translating gs:// URIs to GCS FUSE paths."""
    if model_arg.startswith("gs://"):
        return _gs_uri_to_fuse_path(model_arg)
    return Path(model_arg)


def parse_args(
    argv: list[str] | None = None,
    config: Config | None = None,
) -> argparse.Namespace:
    """Parse CLI arguments for Vertex AI or local dry-run execution."""
    cfg = _cfg(config)
    vertex_cfg = cfg.vertex_ai
    train_cfg = cfg.training

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=train_cfg.epochs)
    parser.add_argument("--batch-size", type=int, default=train_cfg.batch_size)
    parser.add_argument(
        "--training",
        type=str,
        default=os.environ.get(
            "AIP_TRAINING_DATA_URI",
            vertex_cfg.default_training_dir,
        ),
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("AIP_MODEL_DIR", vertex_cfg.default_model_dir),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: auto | mps | cuda | cpu (overrides config)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    try:
        train(
            training_dir=resolve_training_dir(args.training),
            model_dir=resolve_model_dir(args.model_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
    except Exception:
        logger.exception("Training entrypoint failed")
        sys.exit(1)
