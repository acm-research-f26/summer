"""Shared logging configuration for local and Vertex AI training runs."""

from __future__ import annotations

import logging
from pathlib import Path

DEFAULT_LOG_PATH = Path("logs/training_run.log")


def setup_logging(log_path: Path | str = DEFAULT_LOG_PATH) -> logging.Logger:
    """Configure root logger to write to console and a persistent log file."""
    resolved_path = Path(log_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("acm_research")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(resolved_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def log_epoch_metrics(
    logger: logging.Logger,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float | None = None,
    learning_rate: float | None = None,
) -> None:
    """Emit a standardized epoch metrics line."""
    parts = [
        f"Epoch {epoch:02d}/{total_epochs:02d}",
        f"Train Loss: {train_loss:.4f}",
    ]
    if val_loss is not None:
        parts.append(f"Val Loss: {val_loss:.4f}")
    if learning_rate is not None:
        parts.append(f"Learning Rate: {learning_rate:.6f}")
    logger.info(" | ".join(parts))
