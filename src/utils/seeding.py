"""
seeding.py
==========
Deterministic seeding for reproducible pHGFN runs.

Call `set_seed(cfg.system.seed)` at the top of every entry point (training
scripts, the integration test, evaluation). Seeds Python `random`, NumPy, and
PyTorch (CPU + all CUDA devices) from one place so results are reproducible.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic_torch: bool = False) -> None:
    """
    Seed all RNGs used in the project.

    Args:
        seed: The integer seed shared across libraries.
        deterministic_torch: If True, force cuDNN into deterministic mode. This
            makes runs bit-reproducible but can slow training and disallows some
            fast kernels, so it defaults to False (we want speed for training and
            only need *statistical* reproducibility).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # torch is imported lazily so utilities that only need numpy/random don't pay
    # the (heavy) torch import cost.
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch not installed in this context — random/numpy seeding still applied.
        pass
