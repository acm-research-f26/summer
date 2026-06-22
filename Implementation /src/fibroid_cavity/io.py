"""Input/output helpers for NIfTI segmentation masks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np


MASK_SUFFIXES = (".nii", ".nii.gz")


def is_nifti_path(path: Path) -> bool:
    """Return True when a path looks like a NIfTI image."""
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in MASK_SUFFIXES)


def find_mask_paths(mask_dir: Path, pattern: str = "*") -> list[Path]:
    """Find NIfTI masks below a directory."""
    candidates: Iterable[Path] = mask_dir.rglob(pattern)
    return sorted(path for path in candidates if path.is_file() and is_nifti_path(path))


def patient_id_from_path(path: Path) -> str:
    """Derive a stable patient id from a mask filename."""
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    return name


def load_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a labeled segmentation mask and voxel spacing from a NIfTI file."""
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata(), dtype=np.int16)
    spacing = tuple(float(value) for value in image.header.get_zooms()[:3])

    if data.ndim != 3:
        raise ValueError(f"Expected a 3D mask at {path}, got shape {data.shape}")

    if len(spacing) != 3 or any(value <= 0 for value in spacing):
        raise ValueError(f"Invalid voxel spacing for {path}: {spacing}")

    return data, spacing
