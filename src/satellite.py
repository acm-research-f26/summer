"""Sentinel-2 tensor extraction via Earth Engine and mock generation for offline dev."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

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
DEFAULT_RAW_DIR = "data/raw_satellite"
DEFAULT_TILE_DIR = "data/image_tiles"

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


def save_raw_satellite_previews(
    tensor: np.ndarray,
    object_id: int,
    raw_output_dir: str | Path,
    latitude: float,
    longitude: float,
    source: str = "mock_sentinel2_poc",
) -> dict[str, str]:
    """Write human-viewable PNG previews and metadata for a satellite tile."""
    if PilImage is None:
        raise ImportError("Pillow is required to export raw satellite previews.")

    raw_dir = Path(raw_output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    red, green, blue, nir, swir = tensor
    rgb = np.stack([red, green, blue], axis=-1)
    rgb_uint8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    rgb_path = raw_dir / f"tile_{object_id}_rgb.png"
    nir_path = raw_dir / f"tile_{object_id}_nir.png"
    swir_path = raw_dir / f"tile_{object_id}_swir.png"
    meta_path = raw_dir / f"tile_{object_id}_metadata.json"

    PilImage.fromarray(rgb_uint8, mode="RGB").save(rgb_path)
    PilImage.fromarray(_tensor_to_uint8_image(nir), mode="L").save(nir_path)
    PilImage.fromarray(_tensor_to_uint8_image(swir), mode="L").save(swir_path)

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
    meta_path.write_text(json.dumps(metadata, indent=2))

    return {
        "rgb": str(rgb_path),
        "nir": str(nir_path),
        "swir": str(swir_path),
        "metadata": str(meta_path),
    }


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


def fetch_sentinel2_tensors(
    manifest_path: str | Path,
    output_dir: str | Path = DEFAULT_TILE_DIR,
    raw_output_dir: str | Path = DEFAULT_RAW_DIR,
    project_id: str = "datacenter-summer-poc",
    limit: int | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> int:
    """Extract real Sentinel-2 5-channel tiles via Earth Engine per manifest row."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

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
            np.save(output / f"tile_{obj_id}.npy", tensor)
            save_raw_satellite_previews(
                tensor,
                obj_id,
                raw_output_dir,
                latitude=latitude,
                longitude=longitude,
                source="gee_sentinel2_l2a",
            )
            success_count += 1
            if success_count % 10 == 0:
                print(f"Extracted {success_count}/{len(manifest)} tiles...")
        except Exception:
            logger.exception("Failed to extract tile for OBJECTID %d", obj_id)

    print(
        f"Extracted {success_count}/{len(manifest)} Sentinel-2 tiles to {output} "
        f"with previews in {raw_output_dir}"
    )
    return success_count


def generate_mock_satellite_tensors(
    manifest_path: str | Path,
    output_dir: str | Path = DEFAULT_TILE_DIR,
    raw_output_dir: str | Path = DEFAULT_RAW_DIR,
    limit: int | None = None,
) -> int:
    """Generate mock 5-channel float32 tiles and raw PNG previews per OBJECTID."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    if limit is not None:
        manifest = manifest.head(limit)

    for _, row in manifest.iterrows():
        obj_id = int(row["OBJECTID"])
        mock_tensor = _generate_mock_tensor(obj_id)
        np.save(output / f"tile_{obj_id}.npy", mock_tensor)
        save_raw_satellite_previews(
            mock_tensor,
            obj_id,
            raw_output_dir,
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
        )

    count = len(manifest)
    print(
        f"Generated {count} mock multi-spectral tiles in {output} "
        f"and raw previews in {raw_output_dir}"
    )
    return count
