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
    from config import Config, load_config
    from gcs import (
        clear_satellite_imagery,
        prune_orphan_satellite_tiles,
        upload_json,
        upload_npy,
        upload_png,
    )
except ImportError:
    from src.config import Config, load_config
    from src.gcs import (
        clear_satellite_imagery,
        prune_orphan_satellite_tiles,
        upload_json,
        upload_npy,
        upload_png,
    )

if TYPE_CHECKING:
    from PIL import Image as PilImage
else:
    try:
        from PIL import Image as PilImage
    except ImportError:  # pragma: no cover
        PilImage = None

logger = logging.getLogger(__name__)


def _cfg(config: Config | None) -> Config:
    return config or load_config()


def _generate_mock_tensor(object_id: int, config: Config) -> np.ndarray:
    """Create a reproducible mock 5-channel tile for a building OBJECTID."""
    rng = np.random.default_rng(object_id)
    tile_shape = config.dataset.tile_shape
    return rng.random(tile_shape, dtype=np.float32)


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
    source: str | None = None,
    bucket: str | None = None,
    project_id: str | None = None,
    config: Config | None = None,
) -> dict[str, str]:
    """Upload human-viewable PNG previews and metadata for a satellite tile to GCS."""
    cfg = _cfg(config)
    ee_cfg = cfg.earth_engine
    dataset_cfg = cfg.dataset
    resolved_source = source or ee_cfg.source_id

    red, green, blue, nir, swir = tensor
    rgb = np.stack([red, green, blue], axis=-1)
    rgb_uint8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

    upload_kwargs: dict[str, Any] = {"config": cfg}
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
        "tile_shape": list(dataset_cfg.tile_shape),
        "bands": list(dataset_cfg.band_names),
        "footprint_m": cfg.tile_footprint_m,
        "resolution_m": ee_cfg.scale_m,
        "source": resolved_source,
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
    config: Config,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Any:
    import ee

    ee_cfg = config.earth_engine
    collection = (
        ee.ImageCollection(ee_cfg.s2_collection)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", ee_cfg.max_cloud_cover))
        .filterDate(
            start_date or ee_cfg.start_date,
            end_date or ee_cfg.end_date,
        )
    )
    return collection.median().select(list(ee_cfg.band_names)).reproject(
        crs=ee_cfg.reproject_crs,
        scale=ee_cfg.scale_m,
    )


def _arrays_to_tensor(
    band_arrays: dict[str, list[list[float]]],
    config: Config,
) -> np.ndarray:
    """Stack EE band arrays into a normalized (5, H, W) float32 tensor."""
    ee_cfg = config.earth_engine
    tile_shape = config.dataset.tile_shape
    channels = []
    for band in ee_cfg.band_names:
        arr = np.array(band_arrays[band], dtype=np.float32)
        normalized = np.clip(arr / ee_cfg.reflectance_scale, 0.0, 1.0)
        channels.append(normalized)

    tensor = np.stack(channels, axis=0)
    if tensor.shape != tile_shape:
        resized = np.zeros(tile_shape, dtype=np.float32)
        min_h = min(tensor.shape[1], tile_shape[1])
        min_w = min(tensor.shape[2], tile_shape[2])
        resized[:, :min_h, :min_w] = tensor[:, :min_h, :min_w]
        tensor = resized
    return tensor.astype(np.float32)


def _extract_tile(
    composite: Any,
    longitude: float,
    latitude: float,
    config: Config,
) -> np.ndarray:
    import ee

    half_extent_m = config.tile_half_extent_m
    region = ee.Geometry.Point([longitude, latitude]).buffer(half_extent_m).bounds()
    sample = composite.sampleRectangle(region=region, defaultValue=0)
    band_data = sample.getInfo()["properties"]
    return _arrays_to_tensor(band_data, config)


def _upload_tile_bundle(
    tensor: np.ndarray,
    object_id: int,
    latitude: float,
    longitude: float,
    source: str,
    bucket: str | None,
    project_id: str | None,
    config: Config,
) -> None:
    upload_kwargs: dict[str, Any] = {"config": config}
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
        config=config,
    )


def fetch_sentinel2_tensors(
    manifest_path: str | Path,
    project_id: str | None = None,
    bucket: str | None = None,
    limit: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    clear_existing: bool | None = None,
    config: Config | None = None,
) -> int:
    """Extract Sentinel-2 tiles via Earth Engine and upload directly to GCS."""
    cfg = _cfg(config)
    resolved_project = project_id or cfg.gcp.project_id

    manifest = pd.read_csv(manifest_path)
    if limit is not None:
        manifest = manifest.head(limit)
    allowed_ids = set(manifest["OBJECTID"].astype(int))

    should_clear = (
        clear_existing
        if clear_existing is not None
        else cfg.pipeline.clear_satellite_gcs_before_extract
    )
    if should_clear:
        clear_satellite_imagery(
            bucket=bucket, project_id=resolved_project, config=cfg
        )

    _initialize_earth_engine(resolved_project)
    composite = _build_composite(cfg, start_date=start_date, end_date=end_date)

    success_count = 0
    ee_cfg = cfg.earth_engine
    for _, row in manifest.iterrows():
        obj_id = int(row["OBJECTID"])
        latitude = float(row["latitude"])
        longitude = float(row["longitude"])
        try:
            tensor = _extract_tile(composite, longitude, latitude, cfg)
            _upload_tile_bundle(
                tensor,
                obj_id,
                latitude,
                longitude,
                source=ee_cfg.source_id,
                bucket=bucket,
                project_id=project_id,
                config=cfg,
            )
            success_count += 1
            if success_count % ee_cfg.progress_log_every == 0:
                print(f"Extracted {success_count}/{len(manifest)} tiles to GCS...")
        except Exception:
            logger.exception("Failed to extract tile for OBJECTID %d", obj_id)

    prune_orphan_satellite_tiles(
        allowed_ids,
        bucket=bucket,
        project_id=resolved_project,
        config=cfg,
    )
    print(f"Uploaded {success_count}/{len(manifest)} Sentinel-2 tiles to GCS")
    return success_count


def generate_mock_satellite_tensors(
    manifest_path: str | Path,
    bucket: str | None = None,
    project_id: str | None = None,
    limit: int | None = None,
    clear_existing: bool | None = None,
    config: Config | None = None,
) -> int:
    """Generate mock tiles and upload directly to GCS for offline development."""
    cfg = _cfg(config)
    resolved_project = project_id or cfg.gcp.project_id
    manifest = pd.read_csv(manifest_path)
    if limit is not None:
        manifest = manifest.head(limit)
    allowed_ids = set(manifest["OBJECTID"].astype(int))

    should_clear = (
        clear_existing
        if clear_existing is not None
        else cfg.pipeline.clear_satellite_gcs_before_extract
    )
    if should_clear:
        clear_satellite_imagery(
            bucket=bucket, project_id=resolved_project, config=cfg
        )

    for _, row in manifest.iterrows():
        obj_id = int(row["OBJECTID"])
        mock_tensor = _generate_mock_tensor(obj_id, cfg)
        _upload_tile_bundle(
            mock_tensor,
            obj_id,
            float(row["latitude"]),
            float(row["longitude"]),
            source=cfg.earth_engine.mock_source_id,
            bucket=bucket,
            project_id=project_id,
            config=cfg,
        )

    prune_orphan_satellite_tiles(
        allowed_ids,
        bucket=bucket,
        project_id=resolved_project,
        config=cfg,
    )
    count = len(manifest)
    print(f"Uploaded {count} mock multi-spectral tiles to GCS")
    return count
