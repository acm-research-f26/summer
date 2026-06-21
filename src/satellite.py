"""Sentinel-2 tensor extraction via Earth Engine with direct GCS upload."""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

try:
    from gcs import upload_json, upload_npy, upload_png
except ImportError:
    from src.gcs import upload_json, upload_npy, upload_png

if TYPE_CHECKING:
    from PIL import Image as PilImage
else:
    try:
        from PIL import Image as PilImage
    except ImportError:  # pragma: no cover
        PilImage = None

logger = logging.getLogger(__name__)

TILE_SHAPE = (5, 128, 128)
BAND_NAMES = ("red", "green", "blue", "nir", "swir")

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
BAND_NAMES_EE = ["B4", "B3", "B2", "B8", "B11"]
SCALE_M = 10
TILE_PX = 128
HALF_EXTENT_M = (TILE_PX * SCALE_M) / 2
REFLECTANCE_SCALE = 10_000.0
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = "2026-01-01"
MAX_CLOUD_COVER = 5


def _generate_mock_tensor(object_id: int) -> np.ndarray:
    """Create a reproducible mock 5-channel tile for a building OBJECTID."""
    rng = np.random.default_rng(object_id)
    return rng.random(TILE_SHAPE, dtype=np.float32)


def _tensor_to_uint8_image(channel: np.ndarray) -> np.ndarray:
    """Scale a normalized float channel to an 8-bit grayscale image."""
    clipped = np.clip(channel, 0.0, 1.0)
    return (clipped * 255).astype(np.uint8)


def _image_to_png_bytes(image: np.ndarray, mode: str) -> bytes:
    if PilImage is None:
        raise ImportError("Pillow is required to export raw satellite previews.")
    buffer = io.BytesIO()
    PilImage.fromarray(image, mode=mode).save(buffer, format="PNG")
    return buffer.getvalue()


def upload_satellite_previews(
    tensor: np.ndarray,
    object_id: int,
    latitude: float,
    longitude: float,
    source: str = "gee_sentinel2_l2a",
    bucket: str | None = None,
    project_id: str | None = None,
) -> dict[str, str]:
    """Upload human-viewable PNG previews and metadata for a satellite tile to GCS."""
    red, green, blue, nir, swir = tensor
    rgb = np.stack([red, green, blue], axis=-1)
    rgb_uint8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    upload_kwargs: dict[str, Any] = {}
    if bucket is not None:
        upload_kwargs["bucket"] = bucket
    if project_id is not None:
        upload_kwargs["project_id"] = project_id

    rgb_uri = upload_png(
        object_id, "rgb", _image_to_png_bytes(rgb_uint8, "RGB"), **upload_kwargs
    )
    nir_uri = upload_png(
        object_id,
        "nir",
        _image_to_png_bytes(_tensor_to_uint8_image(nir), "L"),
        **upload_kwargs,
    )
    swir_uri = upload_png(
        object_id,
        "swir",
        _image_to_png_bytes(_tensor_to_uint8_image(swir), "L"),
        **upload_kwargs,
    )

    metadata = {
        "OBJECTID": object_id,
        "latitude": latitude,
        "longitude": longitude,
        "tile_shape": list(TILE_SHAPE),
        "bands": list(BAND_NAMES),
        "footprint_m": 1280,
        "resolution_m": 10,
        "source": source,
    }
    meta_uri = upload_json(
        object_id,
        json.dumps(metadata, indent=2),
        **upload_kwargs,
    )

    return {"rgb": rgb_uri, "nir": nir_uri, "swir": swir_uri, "metadata": meta_uri}


def _initialize_earth_engine(project_id: str) -> None:
    import ee

    ee.Initialize(project=project_id)


def _build_composite(
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> Any:
    import ee

    collection = (
        ee.ImageCollection(S2_COLLECTION)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_COVER))
        .filterDate(start_date, end_date)
    )
    return collection.median().select(BAND_NAMES_EE)


def _arrays_to_tensor(band_arrays: dict[str, list[list[float]]]) -> np.ndarray:
    """Stack EE band arrays into a normalized (5, H, W) float32 tensor."""
    channels = []
    for band in BAND_NAMES_EE:
        arr = np.array(band_arrays[band], dtype=np.float32)
        normalized = np.clip(arr / REFLECTANCE_SCALE, 0.0, 1.0)
        channels.append(normalized)

    tensor = np.stack(channels, axis=0)
    if tensor.shape != TILE_SHAPE:
        resized = np.zeros(TILE_SHAPE, dtype=np.float32)
        min_h = min(tensor.shape[1], TILE_SHAPE[1])
        min_w = min(tensor.shape[2], TILE_SHAPE[2])
        resized[:, :min_h, :min_w] = tensor[:, :min_h, :min_w]
        tensor = resized
    return tensor.astype(np.float32)


def _extract_tile(composite: Any, longitude: float, latitude: float) -> np.ndarray:
    import ee

    region = ee.Geometry.Point([longitude, latitude]).buffer(HALF_EXTENT_M).bounds()
    sample = composite.sampleRectangle(region=region, defaultValue=0)
    band_data = sample.getInfo()["properties"]
    return _arrays_to_tensor(band_data)


def _upload_tile_bundle(
    tensor: np.ndarray,
    object_id: int,
    latitude: float,
    longitude: float,
    source: str,
    bucket: str | None,
    project_id: str | None,
) -> None:
    upload_kwargs: dict[str, Any] = {}
    if bucket is not None:
        upload_kwargs["bucket"] = bucket
    if project_id is not None:
        upload_kwargs["project_id"] = project_id

    upload_npy(object_id, tensor, **upload_kwargs)
    upload_satellite_previews(
        tensor,
        object_id,
        latitude=latitude,
        longitude=longitude,
        source=source,
        bucket=bucket,
        project_id=project_id,
    )


def fetch_sentinel2_tensors(
    manifest_path: str | Path,
    project_id: str = "datacenter-summer-poc",
    bucket: str | None = None,
    limit: int | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> int:
    """Extract Sentinel-2 tiles via Earth Engine and upload directly to GCS."""
    _initialize_earth_engine(project_id)
    composite = _build_composite(start_date=start_date, end_date=end_date)

    manifest = pd.read_csv(manifest_path)
    if limit is not None:
        manifest = manifest.head(limit)

    success_count = 0
    for _, row in manifest.iterrows():
        obj_id = int(row["OBJECTID"])
        latitude = float(row["latitude"])
        longitude = float(row["longitude"])
        try:
            tensor = _extract_tile(composite, longitude, latitude)
            _upload_tile_bundle(
                tensor,
                obj_id,
                latitude,
                longitude,
                source="gee_sentinel2_l2a",
                bucket=bucket,
                project_id=project_id,
            )
            success_count += 1
            if success_count % 10 == 0:
                print(f"Extracted {success_count}/{len(manifest)} tiles to GCS...")
        except Exception:
            logger.exception("Failed to extract tile for OBJECTID %d", obj_id)

    print(f"Uploaded {success_count}/{len(manifest)} Sentinel-2 tiles to GCS")
    return success_count


def generate_mock_satellite_tensors(
    manifest_path: str | Path,
    bucket: str | None = None,
    project_id: str | None = None,
    limit: int | None = None,
) -> int:
    """Generate mock tiles and upload directly to GCS for offline development."""
    manifest = pd.read_csv(manifest_path)
    if limit is not None:
        manifest = manifest.head(limit)

    for _, row in manifest.iterrows():
        obj_id = int(row["OBJECTID"])
        mock_tensor = _generate_mock_tensor(obj_id)
        _upload_tile_bundle(
            mock_tensor,
            obj_id,
            float(row["latitude"]),
            float(row["longitude"]),
            source="mock_sentinel2_poc",
            bucket=bucket,
            project_id=project_id,
        )

    count = len(manifest)
    print(f"Uploaded {count} mock multi-spectral tiles to GCS")
    return count
