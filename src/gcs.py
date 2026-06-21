"""Google Cloud Storage helpers for bucket-root imagery and training artifacts."""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
from google.cloud import storage

GCS_BUCKET = "datacenter-summer-poc-data"
GCS_TILES_PREFIX = "image_tiles"
GCS_RAW_PREFIX = "raw_satellite"
GCS_MANIFEST_BLOB = "parsed_manifest.csv"
GCS_OUTPUT_PREFIX = "output/models"
GCS_FUSE_ROOT = f"/gcs/{GCS_BUCKET}"


def _client(project_id: str | None = None) -> storage.Client:
    return storage.Client(project=project_id)


def _bucket(bucket_name: str, project_id: str | None = None) -> storage.Bucket:
    return _client(project_id).bucket(bucket_name)


def tile_blob_name(object_id: int) -> str:
    return f"{GCS_TILES_PREFIX}/tile_{object_id}.npy"


def raw_blob_name(object_id: int, band: str) -> str:
    return f"{GCS_RAW_PREFIX}/tile_{object_id}_{band}.png"


def metadata_blob_name(object_id: int) -> str:
    return f"{GCS_RAW_PREFIX}/tile_{object_id}_metadata.json"


def upload_bytes(
    blob_name: str,
    data: bytes,
    bucket: str = GCS_BUCKET,
    content_type: str | None = None,
    project_id: str | None = None,
) -> str:
    """Upload raw bytes to a GCS blob and return its gs:// URI."""
    blob = _bucket(bucket, project_id).blob(blob_name)
    blob.upload_from_string(data, content_type=content_type)
    return f"gs://{bucket}/{blob_name}"


def upload_npy(
    object_id: int,
    tensor: np.ndarray,
    bucket: str = GCS_BUCKET,
    project_id: str | None = None,
) -> str:
    """Upload a float32 tile tensor to image_tiles/."""
    buffer = io.BytesIO()
    np.save(buffer, tensor.astype(np.float32))
    return upload_bytes(
        tile_blob_name(object_id),
        buffer.getvalue(),
        bucket=bucket,
        content_type="application/octet-stream",
        project_id=project_id,
    )


def upload_png(
    object_id: int,
    band: str,
    image_bytes: bytes,
    bucket: str = GCS_BUCKET,
    project_id: str | None = None,
) -> str:
    """Upload a PNG preview to raw_satellite/."""
    return upload_bytes(
        raw_blob_name(object_id, band),
        image_bytes,
        bucket=bucket,
        content_type="image/png",
        project_id=project_id,
    )


def upload_json(
    object_id: int,
    payload: str,
    bucket: str = GCS_BUCKET,
    project_id: str | None = None,
) -> str:
    """Upload tile metadata JSON to raw_satellite/."""
    return upload_bytes(
        metadata_blob_name(object_id),
        payload.encode("utf-8"),
        bucket=bucket,
        content_type="application/json",
        project_id=project_id,
    )


def upload_manifest(
    local_path: str | Path,
    bucket: str = GCS_BUCKET,
    project_id: str | None = None,
) -> str:
    """Upload parsed_manifest.csv to the bucket root."""
    data = Path(local_path).read_bytes()
    uri = upload_bytes(
        GCS_MANIFEST_BLOB,
        data,
        bucket=bucket,
        content_type="text/csv",
        project_id=project_id,
    )
    print(f"Manifest uploaded to {uri}")
    return uri


def blob_exists(
    blob_name: str,
    bucket: str = GCS_BUCKET,
    project_id: str | None = None,
) -> bool:
    """Return True if a blob exists in the bucket."""
    return _bucket(bucket, project_id).blob(blob_name).exists()


def download_blob(
    blob_name: str,
    bucket: str = GCS_BUCKET,
    project_id: str | None = None,
) -> bytes:
    """Download a blob's contents as bytes."""
    return _bucket(bucket, project_id).blob(blob_name).download_as_bytes()


def list_tile_object_ids(
    bucket: str = GCS_BUCKET,
    project_id: str | None = None,
) -> set[int]:
    """List OBJECTIDs present in image_tiles/."""
    client = _client(project_id)
    prefix = f"{GCS_TILES_PREFIX}/tile_"
    object_ids: set[int] = set()
    for blob in client.list_blobs(bucket, prefix=prefix):
        match = re.search(r"tile_(\d+)\.npy$", blob.name)
        if match:
            object_ids.add(int(match.group(1)))
    return object_ids


def get_latest_run_prefix(
    bucket: str = GCS_BUCKET,
    project_id: str | None = None,
) -> str | None:
    """Return the newest output/models/{run_id}/ prefix, or None."""
    client = _client(project_id)
    runs: set[str] = set()
    for blob in client.list_blobs(bucket, prefix=f"{GCS_OUTPUT_PREFIX}/"):
        parts = blob.name.split("/")
        if len(parts) >= 3 and parts[2]:
            runs.add(parts[2])
    if not runs:
        return None
    latest = sorted(runs)[-1]
    return f"{GCS_OUTPUT_PREFIX}/{latest}/"


def download_artifact_prefix(
    run_prefix: str,
    local_dir: str | Path,
    bucket: str = GCS_BUCKET,
    project_id: str | None = None,
) -> Path:
    """Download all blobs under a run prefix to a local directory."""
    output = Path(local_dir)
    output.mkdir(parents=True, exist_ok=True)
    client = _client(project_id)
    prefix = run_prefix.rstrip("/") + "/"
    count = 0
    for blob in client.list_blobs(bucket, prefix=prefix):
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
    bucket: str = GCS_BUCKET,
    project_id: str | None = None,
) -> Path:
    """Download manifest and image tiles from bucket root for local dry-runs."""
    data_path = Path(local_data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    client = _client(project_id)

    manifest_blob = client.bucket(bucket).blob(GCS_MANIFEST_BLOB)
    if manifest_blob.exists():
        manifest_blob.download_to_filename(str(data_path / GCS_MANIFEST_BLOB))

    tiles_dir = data_path / GCS_TILES_PREFIX
    tiles_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for blob in client.list_blobs(bucket, prefix=f"{GCS_TILES_PREFIX}/"):
        if blob.name.endswith("/"):
            continue
        relative = Path(blob.name).name
        blob.download_to_filename(str(tiles_dir / relative))
        count += 1

    print(f"Downloaded manifest and {count} tiles to {data_path}")
    return data_path
