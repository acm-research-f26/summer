"""Plotting configuration shared by reporting modules."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def configure_matplotlib() -> None:
    """Use writable cache directories and a non-GUI backend."""
    cache_root = Path(tempfile.gettempdir()) / "fibroid_cavity_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
