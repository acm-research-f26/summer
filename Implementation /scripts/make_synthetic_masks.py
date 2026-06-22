#!/usr/bin/env python3
"""Create synthetic NIfTI masks for end-to-end pipeline demos."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from fibroid_cavity.constants import CAVITY_LABEL, FIBROID_LABEL, UTERINE_WALL_LABEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/demo/raw"), help="Output mask directory.")
    parser.add_argument("--patients", type=int, default=30, help="Number of synthetic patient masks to create.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for patient_idx in range(args.patients):
        mask = _make_patient_mask(patient_idx, rng)
        image = nib.Nifti1Image(mask.astype(np.int16), affine=np.diag([1.25, 1.25, 3.0, 1.0]))
        image.header.set_zooms((1.25, 1.25, 3.0))
        nib.save(image, args.output_dir / f"synthetic_patient_{patient_idx:03d}.nii.gz")

    print(f"Wrote {args.patients} synthetic masks to {args.output_dir}")


def _make_patient_mask(patient_idx: int, rng: np.random.Generator) -> np.ndarray:
    shape = (48, 48, 24)
    mask = np.zeros(shape, dtype=np.int16)
    coords = np.indices(shape).astype(float)

    center = np.array([24.0 + rng.normal(0, 0.5), 24.0 + rng.normal(0, 0.5), 12.0])
    wall_radii = np.array([17.0, 18.0, 8.0])
    cavity_radii = np.array([5.5, 6.5, 2.8])

    wall = _ellipsoid(coords, center, wall_radii)
    cavity = _ellipsoid(coords, center, cavity_radii)
    mask[wall] = UTERINE_WALL_LABEL
    mask[cavity] = CAVITY_LABEL

    cavity_touching = patient_idx % 2 == 0
    fibroid_radius = int(rng.integers(2, 5))
    fibroid_center = _fibroid_center(center, cavity_touching, fibroid_radius, rng)
    fibroid = _sphere(coords, fibroid_center, fibroid_radius)
    fibroid = np.logical_and(fibroid, ~cavity)
    mask[fibroid] = FIBROID_LABEL

    if patient_idx % 5 == 0:
        secondary_center = np.array([10.0 + rng.normal(0, 1), 36.0 + rng.normal(0, 1), 9.0])
        secondary = _sphere(coords, secondary_center, 2)
        secondary = np.logical_and(secondary, ~cavity)
        mask[secondary] = FIBROID_LABEL

    return mask


def _fibroid_center(
    cavity_center: np.ndarray,
    cavity_touching: bool,
    fibroid_radius: int,
    rng: np.random.Generator,
) -> np.ndarray:
    direction = rng.choice([-1.0, 1.0])
    y_offset = rng.normal(0, 1.0)
    z_offset = rng.normal(0, 0.5)

    if cavity_touching:
        x_offset = direction * (5.5 + fibroid_radius + 0.4)
    else:
        x_offset = direction * (13.0 + fibroid_radius + rng.uniform(1.5, 4.0))

    return cavity_center + np.array([x_offset, y_offset, z_offset])


def _ellipsoid(coords: np.ndarray, center: np.ndarray, radii: np.ndarray) -> np.ndarray:
    scaled = ((coords[0] - center[0]) / radii[0]) ** 2
    scaled += ((coords[1] - center[1]) / radii[1]) ** 2
    scaled += ((coords[2] - center[2]) / radii[2]) ** 2
    return scaled <= 1.0


def _sphere(coords: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    dist = (coords[0] - center[0]) ** 2
    dist += (coords[1] - center[1]) ** 2
    dist += (coords[2] - center[2]) ** 2
    return dist <= radius**2


if __name__ == "__main__":
    main()
