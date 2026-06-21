"""Vertex AI training entrypoint for the data center vision classifier."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from logging_config import log_epoch_metrics, setup_logging
from model_def import DataCenterVisionNet

logger = setup_logging()

DEFAULT_GCS_FUSE_TRAINING = "/gcs/datacenter-summer-poc-data/input/data-center-vision"


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


def compute_class_weights(manifest: pd.DataFrame, num_classes: int = 3) -> torch.Tensor:
    """Compute inverse-frequency class weights normalized to num_classes."""
    counts = (
        manifest["target_label"]
        .value_counts()
        .reindex(range(num_classes), fill_value=1)
    )
    weights = 1.0 / counts.astype(float)
    normalized = weights / weights.sum() * num_classes
    return torch.tensor(normalized.values, dtype=torch.float32)


def stratified_train_val_indices(
    manifest: pd.DataFrame,
    val_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[list[int], list[int]]:
    """Split manifest indices into stratified train and validation sets."""
    rng = np.random.default_rng(random_state)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for label in sorted(manifest["target_label"].unique()):
        label_rows = manifest[manifest["target_label"] == label]
        indices = label_rows.index.to_numpy(copy=True)
        rng.shuffle(indices)
        n_val = int(len(indices) * val_fraction)
        n_val = max(1, n_val) if len(indices) > 1 else 0
        val_indices.extend(indices[:n_val].tolist())
        train_indices.extend(indices[n_val:].tolist())

    return train_indices, val_indices


def _run_epoch(
    model: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    epoch_loss = 0.0
    batch_count = 0

    for imgs, labels in dataloader:
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


def train(
    training_dir: str | Path,
    model_dir: str | Path,
    epochs: int,
    batch_size: int,
    learning_rate: float = 0.001,
    val_fraction: float = 0.2,
) -> None:
    """Run the Vertex AI training loop and persist model weights."""
    training_path = Path(training_dir)
    csv_path = training_path / "parsed_manifest.csv"
    image_path = training_path / "image_tiles"

    manifest = pd.read_csv(csv_path)
    class_counts = manifest["target_label"].value_counts().sort_index()
    logger.info("Class distribution: %s", class_counts.to_dict())

    full_dataset = VisionImageDataset(csv_path, image_path)
    train_idx, val_idx = stratified_train_val_indices(
        manifest,
        val_fraction=val_fraction,
    )
    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        Subset(full_dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
    )

    model = DataCenterVisionNet(num_classes=3)
    class_weights = compute_class_weights(manifest)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    logger.info(
        "Beginning Vertex AI training loop (%d train / %d val samples)...",
        len(train_idx),
        len(val_idx),
    )
    for epoch in range(epochs):
        try:
            train_loss = _run_epoch(model, train_loader, criterion, optimizer)
            val_loss = _run_epoch(model, val_loader, criterion, optimizer=None)
            current_lr = optimizer.param_groups[0]["lr"]
            log_epoch_metrics(
                logger,
                epoch=epoch + 1,
                total_epochs=epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
            )
        except Exception:
            logger.exception("Training failed during epoch %d", epoch + 1)
            raise

    model_output_path = Path(model_dir) / "model.pth"
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_output_path)
    logger.info("Model weights saved to %s", model_output_path)


def resolve_training_dir(training_arg: str) -> Path:
    """Resolve training directory from CLI arg, GCS fuse mount, or local fallback."""
    candidate = Path(training_arg)
    if candidate.exists():
        return candidate
    fuse_path = Path(DEFAULT_GCS_FUSE_TRAINING)
    if fuse_path.exists():
        return fuse_path
    return candidate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for Vertex AI or local dry-run execution."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--training",
        type=str,
        default=os.environ.get("AIP_TRAINING_DATA_URI", "data"),
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("AIP_MODEL_DIR", "/tmp/model"),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    try:
        train(
            training_dir=resolve_training_dir(args.training),
            model_dir=args.model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    except Exception:
        logger.exception("Vertex AI entrypoint failed")
        sys.exit(1)
