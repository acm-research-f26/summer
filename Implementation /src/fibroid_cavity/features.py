"""Geometry extraction from labeled uterine MRI masks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi

from fibroid_cavity.constants import CAVITY_LABEL, FIBROID_LABEL


@dataclass(frozen=True)
class FibroidFeatures:
    """Feature row for one connected fibroid component."""

    patient_id: str
    fibroid_id: int
    volume_voxels: int
    volume_mm3: float
    centroid_x: float
    centroid_y: float
    centroid_z: float
    cavity_centroid_x: float
    cavity_centroid_y: float
    cavity_centroid_z: float
    centroid_to_cavity_dist_mm: float
    bbox_size_x_mm: float
    bbox_size_y_mm: float
    bbox_size_z_mm: float
    aspect_ratio: float
    boundary_contact_count: int
    boundary_contact_ratio: float
    overlap_count: int
    overlap_ratio: float
    min_distance_to_cavity_mm: float
    cavity_touching: int


def extract_patient_features(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    patient_id: str,
    contact_iterations: int = 1,
) -> list[dict[str, object]]:
    """Extract fibroid-level features from one labeled segmentation mask."""
    if mask.ndim != 3:
        raise ValueError(f"Expected a 3D mask, got shape {mask.shape}")

    fibroid_mask = mask == FIBROID_LABEL
    cavity_mask = mask == CAVITY_LABEL

    if not fibroid_mask.any():
        return []

    labeled_fibroids, component_count = ndi.label(fibroid_mask, structure=np.ones((3, 3, 3), dtype=bool))
    cavity_centroid_vox = _centroid(cavity_mask)
    cavity_centroid_mm = _scale_point(cavity_centroid_vox, spacing)
    cavity_boundary = _boundary(cavity_mask)
    cavity_distance_mm = _distance_to_mask_mm(cavity_mask, spacing)

    rows: list[dict[str, object]] = []
    voxel_volume_mm3 = float(np.prod(spacing))

    for fibroid_id in range(1, component_count + 1):
        component = labeled_fibroids == fibroid_id
        volume_voxels = int(component.sum())
        if volume_voxels == 0:
            continue

        centroid_vox = _centroid(component)
        centroid_mm = _scale_point(centroid_vox, spacing)
        centroid_to_cavity = _euclidean_mm(centroid_mm, cavity_centroid_mm)
        bbox_sizes_mm = _bbox_sizes_mm(component, spacing)
        aspect_ratio = _aspect_ratio(bbox_sizes_mm)

        dilated_component = ndi.binary_dilation(component, iterations=contact_iterations)
        boundary_contact_count = int(np.logical_and(dilated_component, cavity_boundary).sum())
        boundary_contact_ratio = boundary_contact_count / volume_voxels

        overlap_count = int(np.logical_and(component, cavity_mask).sum())
        overlap_ratio = overlap_count / volume_voxels

        distances = cavity_distance_mm[component]
        min_distance = float(distances.min()) if distances.size else float("nan")
        cavity_touching = int(boundary_contact_count > 0 or overlap_count > 0)

        row = FibroidFeatures(
            patient_id=patient_id,
            fibroid_id=fibroid_id,
            volume_voxels=volume_voxels,
            volume_mm3=volume_voxels * voxel_volume_mm3,
            centroid_x=centroid_mm[0],
            centroid_y=centroid_mm[1],
            centroid_z=centroid_mm[2],
            cavity_centroid_x=cavity_centroid_mm[0],
            cavity_centroid_y=cavity_centroid_mm[1],
            cavity_centroid_z=cavity_centroid_mm[2],
            centroid_to_cavity_dist_mm=centroid_to_cavity,
            bbox_size_x_mm=bbox_sizes_mm[0],
            bbox_size_y_mm=bbox_sizes_mm[1],
            bbox_size_z_mm=bbox_sizes_mm[2],
            aspect_ratio=aspect_ratio,
            boundary_contact_count=boundary_contact_count,
            boundary_contact_ratio=boundary_contact_ratio,
            overlap_count=overlap_count,
            overlap_ratio=overlap_ratio,
            min_distance_to_cavity_mm=min_distance,
            cavity_touching=cavity_touching,
        )
        rows.append(asdict(row))

    return rows


def _centroid(mask: np.ndarray) -> tuple[float, float, float]:
    if not mask.any():
        return (float("nan"), float("nan"), float("nan"))
    coords = np.argwhere(mask)
    values = coords.mean(axis=0)
    return (float(values[0]), float(values[1]), float(values[2]))


def _scale_point(
    point: tuple[float, float, float],
    spacing: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(float(coord * voxel_size) for coord, voxel_size in zip(point, spacing))


def _euclidean_mm(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    if any(np.isnan(value) for value in first + second):
        return float("nan")
    return float(np.linalg.norm(np.asarray(first) - np.asarray(second)))


def _boundary(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    eroded = ndi.binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    return np.logical_and(mask, ~eroded)


def _distance_to_mask_mm(mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    if not mask.any():
        return np.full(mask.shape, np.inf, dtype=float)
    return ndi.distance_transform_edt(~mask, sampling=spacing)


def _bbox_sizes_mm(component: np.ndarray, spacing: tuple[float, float, float]) -> tuple[float, float, float]:
    coords = np.argwhere(component)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    sizes_voxels = (maxs - mins + 1).astype(float)
    sizes_mm = sizes_voxels * np.asarray(spacing, dtype=float)
    return (float(sizes_mm[0]), float(sizes_mm[1]), float(sizes_mm[2]))


def _aspect_ratio(bbox_sizes_mm: tuple[float, float, float]) -> float:
    sizes = np.asarray([size for size in bbox_sizes_mm if size > 0], dtype=float)
    if sizes.size == 0:
        return float("nan")
    return float(sizes.max() / sizes.min())
