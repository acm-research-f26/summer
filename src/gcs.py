"""Google Cloud Storage helpers for bucket-root imagery and training artifacts."""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
from google.cloud import storage

try:
    from config import Config, load_config
except ImportError:
    from src.config import Config, load_config


def _cfg(config: Config | None) -> Config:
    return config or load_config()


def _client(project_id: str | None = None) -> storage.Client:
    return storage.Client(project=project_id)


def _bucket(bucket_name: str, project_id: str | None = None) -> storage.Bucket:
    return _client(project_id).bucket(bucket_name)


def _resolve_gcs(
    config: Config | None,
    bucket: str | None,
    project_id: str | None,
) -> tuple[str, str | None]:
    cfg = _cfg(config)
    return bucket or cfg.gcs.bucket_name, project_id or cfg.gcp.project_id


def tile_blob_name(object_id: int, config: Config | None = None) -> str:
    prefixes = _cfg(config).gcs.prefixes
    return f"{prefixes.image_tiles}/tile_{object_id}.npy"


def raw_blob_name(object_id: int, band: str, config: Config | None = None) -> str:
    prefixes = _cfg(config).gcs.prefixes
    return f"{prefixes.raw_satellite}/tile_{object_id}_{band}.png"


def metadata_blob_name(object_id: int, config: Config | None = None) -> str:
    prefixes = _cfg(config).gcs.prefixes
    return f"{prefixes.raw_satellite}/tile_{object_id}_metadata.json"


def tile_blob_names_for_object(
    object_id: int, config: Config | None = None
) -> list[str]:
    """Return all GCS blob paths for one building tile bundle."""
    return [
        tile_blob_name(object_id, config=config),
        raw_blob_name(object_id, "rgb", config=config),
        raw_blob_name(object_id, "nir", config=config),
        raw_blob_name(object_id, "swir", config=config),
        metadata_blob_name(object_id, config=config),
    ]


def upload_bytes(
    blob_name: str,
    data: bytes,
    bucket: str | None = None,
    content_type: str | None = None,
    project_id: str | None = None,
    config: Config | None = None,
) -> str:
    """Upload raw bytes to a GCS blob, overwriting any existing object."""
    resolved_bucket, resolved_project = _resolve_gcs(config, bucket, project_id)
    blob = _bucket(resolved_bucket, resolved_project).blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{resolved_bucket}/{blob_name}"


def upload_npy(
    object_id: int,
    tensor: np.ndarray,
    bucket: str | None = None,
    project_id: str | None = None,
    config: Config | None = None,
) -> str:
    """Upload a float32 tile tensor to image_tiles/."""
    buffer = io.BytesIO()
    np.save(buffer, tensor.astype(np.float32))
    return upload_bytes(
        tile_blob_name(object_id, config=config),
        buffer.getvalue(),
        bucket=bucket,
        content_type="application/octet-stream",
        project_id=project_id,
        config=config,
    )


def upload_png(
    object_id: int,
    band: str,
    image_bytes: bytes,
    bucket: str | None = None,
    project_id: str | None = None,
    config: Config | None = None,
) -> str:
    """Upload a PNG preview to raw_satellite/."""
    return upload_bytes(
        raw_blob_name(object_id, band, config=config),
        image_bytes,
        bucket=bucket,
        content_type="image/png",
        project_id=project_id,
        config=config,
    )


def upload_json(
    object_id: int,
    payload: str,
    bucket: str | None = None,
    project_id: str | None = None,
    config: Config | None = None,
) -> str:
    """Upload tile metadata JSON to raw_satellite/."""
    return upload_bytes(
        metadata_blob_name(object_id, config=config),
        payload.encode("utf-8"),
        bucket=bucket,
        content_type="application/json",
        project_id=project_id,
        config=config,
    )


def upload_manifest(
    local_path: str | Path,
    bucket: str | None = None,
    project_id: str | None = None,
    config: Config | None = None,
) -> str:
    """Upload parsed_manifest.csv to the bucket root."""
    cfg = _cfg(config)
    data = Path(local_path).read_bytes()
    uri = upload_bytes(
        cfg.gcs.manifest_blob,
        data,
        bucket=bucket,
        content_type="text/csv",
        project_id=project_id,
        config=config,
    )
    print(f"Manifest uploaded to {uri}")
    return uri


def blob_exists(
    blob_name: str,
    bucket: str | None = None,
    project_id: str | None = None,
    config: Config | None = None,
) -> bool:
    """Return True if a blob exists in the bucket."""
    resolved_bucket, resolved_project = _resolve_gcs(config, bucket, project_id)
    return _bucket(resolved_bucket, resolved_project).blob(blob_name).exists()


def download_blob(
    blob_name: str,
    bucket: str | None = None,
    project_id: str | None = None,
    config: Config | None = None,
) -> bytes:
    """Download a blob's contents as bytes."""
    resolved_bucket, resolved_project = _resolve_gcs(config, bucket, project_id)
    blob = _bucket(resolved_bucket, resolved_project).blob(blob_name)
    return blob.download_as_bytes()


def list_tile_object_ids(
    bucket: str | None = None,
    project_id: str | None = None,
    config: Config | None = None,
) -> set[int]:
    """List OBJECTIDs present in image_tiles/."""
    cfg = _cfg(config)
    resolved_bucket, resolved_project = _resolve_gcs(config, bucket, project_id)
    client = _client(resolved_project)
    prefix = f"{cfg.gcs.prefixes.image_tiles}/tile_"
    object_ids: set[int] = set()
    for blob in client.list_blobs(resolved_bucket, prefix=prefix):
        match = re.search(r"tile_(\d+)\.npy$", blob.name)
        if match:
            object_ids.add(int(match.group(1)))
    return object_ids


def clear_satellite_imagery(
    bucket: str | None = None,
    project_id: str | None = None,
    config: Config | None = None,
) -> int:
    """Delete all objects under image_tiles/ and raw_satellite/."""
    cfg = _cfg(config)
    resolved_bucket, resolved_project = _resolve_gcs(config, bucket, project_id)
    client = _client(resolved_project)
    prefixes = (cfg.gcs.prefixes.image_tiles, cfg.gcs.prefixes.raw_satellite)
    deleted = 0
    for prefix in prefixes:
        blobs = list(client.list_blobs(resolved_bucket, prefix=f"{prefix}/"))
        if blobs:
            client.bucket(resolved_bucket).delete_blobs(blobs)
            deleted += len(blobs)
    print(
        f"Cleared {deleted} satellite object(s) from "
        f"gs://{resolved_bucket}/{{{prefixes[0]}, {prefixes[1]}}}/"
    )
    return deleted


def prune_orphan_satellite_tiles(
    allowed_object_ids: set[int],
    bucket: str | None = None,
    project_id: str | None = None,
    config: Config | None = None,
) -> int:
    """Remove tile bundles in GCS that are not in the allowed OBJECTID set."""
    stored_ids = list_tile_object_ids(
        bucket=bucket, project_id=project_id, config=config
    )
    orphan_ids = stored_ids - allowed_object_ids
    if not orphan_ids:
        return 0

    resolved_bucket, resolved_project = _resolve_gcs(config, bucket, project_id)
    bucket_ref = _bucket(resolved_bucket, resolved_project)
    deleted = 0
    for object_id in sorted(orphan_ids):
        for blob_name in tile_blob_names_for_object(object_id, config=config):
            blob = bucket_ref.blob(blob_name)
            if blob.exists():
                blob.delete()
                deleted += 1
    print(f"Pruned {len(orphan_ids)} orphan tile(s) ({deleted} blob(s) deleted)")
    return deleted


def get_latest_run_prefix(
    bucket: str | None = None,
    project_id: str | None = None,
    config: Config | None = None,
) -> str | None:
    """Return the newest output/models/{run_id}/ prefix, or None."""
    cfg = _cfg(config)
    resolved_bucket, resolved_project = _resolve_gcs(config, bucket, project_id)
    output_prefix = cfg.gcs.prefixes.output_models
    client = _client(resolved_project)
    runs: set[str] = set()
    for blob in client.list_blobs(resolved_bucket, prefix=f"{output_prefix}/"):
        parts = blob.name.split("/")
        if len(parts) >= 3 and parts[2]:
            runs.add(parts[2])
    if not runs:
        return None
    latest = sorted(runs)[-1]
    return f"{output_prefix}/{latest}/"


def download_artifact_prefix(
    run_prefix: str,
    local_dir: str | Path,
    bucket: str | None = None,
    project_id: str | None = None,
    config: Config | None = None,
) -> Path:
    """Download all blobs under a run prefix to a local directory."""
    resolved_bucket, resolved_project = _resolve_gcs(config, bucket, project_id)
    output = Path(local_dir)
    output.mkdir(parents=True, exist_ok=True)
    client = _client(resolved_project)
    prefix = run_prefix.rstrip("/") + "/"
    count = 0
    for blob in client.list_blobs(resolved_bucket, prefix=prefix):
        if blob.name.endswith("/"):
            continue
        relative = blob.name[len(prefix) :]
        local_file = output / relative
        local_file.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_file))
        count += 1
    print(f"Downloaded {count} artifacts to {output}")
    return output


def download_training_data(
    local_data_dir: str | Path,
    bucket: str | None = None,
    project_id: str | None = None,
    config: Config | None = None,
) -> Path:
    """Download manifest and image tiles from bucket root for local dry-runs."""
    cfg = _cfg(config)
    resolved_bucket, resolved_project = _resolve_gcs(config, bucket, project_id)
    data_path = Path(local_data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    client = _client(resolved_project)

    manifest_blob = client.bucket(resolved_bucket).blob(cfg.gcs.manifest_blob)
    if manifest_blob.exists():
        manifest_blob.download_to_filename(str(data_path / cfg.gcs.manifest_blob))

    tiles_dir = data_path / cfg.gcs.prefixes.image_tiles
    tiles_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    tiles_prefix = cfg.gcs.prefixes.image_tiles
    for blob in client.list_blobs(resolved_bucket, prefix=f"{tiles_prefix}/"):
        if blob.name.endswith("/"):
            continue
        relative = Path(blob.name).name
        blob.download_to_filename(str(tiles_dir / relative))
        count += 1

    print(f"Downloaded manifest and {count} tiles to {data_path}")
    return data_path


# Backward-compatible aliases derived from the default configuration.
_default = load_config()
GCS_BUCKET = _default.gcs.bucket_name
GCS_TILES_PREFIX = _default.gcs.prefixes.image_tiles
GCS_RAW_PREFIX = _default.gcs.prefixes.raw_satellite
GCS_MANIFEST_BLOB = _default.gcs.manifest_blob
GCS_OUTPUT_PREFIX = _default.gcs.prefixes.output_models
GCS_FUSE_ROOT = _default.gcs.fuse_root
