"""Shared logging configuration for local and Vertex AI training runs."""

from __future__ import annotations

import logging
from pathlib import Path

try:
    from .config import Config, load_config
except ImportError:
    try:
        from config import Config, load_config
    except ImportError:
        from src.config import Config, load_config


def setup_logging(
    log_path: Path | str | None = None,
    config: Config | None = None,
) -> logging.Logger:
    """Configure root logger to write to console and a persistent log file."""
    cfg = config or load_config()
    log_cfg = cfg.logging
    resolved_path = Path(log_path or log_cfg.log_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(log_cfg.logger_name)
    logger.setLevel(getattr(logging, log_cfg.level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        log_cfg.message_format,
        datefmt=log_cfg.date_format,
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
