"""Google Cloud Storage upload and download helpers for training data."""

from __future__ import annotations

from pathlib import Path

from google.cloud import storage

GCS_BUCKET = "datacenter-summer-poc-data"
GCS_INPUT_PREFIX = "input/data-center-vision"
GCS_OUTPUT_PREFIX = "output/models"

UPLOAD_PATHS = (
    "parsed_manifest.csv",
    "image_tiles",
    "raw_satellite",
)


def _upload_file(
    client: storage.Client,
    bucket_name: str,
    local_path: Path,
    blob_name: str,
) -> None:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(local_path))


def _upload_directory(
    client: storage.Client,
    bucket_name: str,
    local_dir: Path,
    prefix: str,
) -> int:
    count = 0
    for file_path in local_dir.rglob("*"):
        if file_path.is_file():
            relative = file_path.relative_to(local_dir)
            blob_name = f"{prefix}/{relative.as_posix()}"
            _upload_file(client, bucket_name, file_path, blob_name)
            count += 1
    return count


def upload_training_data(
    local_data_dir: str | Path,
    bucket: str = GCS_BUCKET,
    prefix: str = GCS_INPUT_PREFIX,
    project_id: str | None = None,
) -> str:
    """Upload manifest, image tiles, and raw previews to GCS.

    Returns the gs:// URI prefix used as Vertex training input.
    """
    data_path = Path(local_data_dir)
    client = storage.Client(project=project_id)
    gcs_prefix = prefix.rstrip("/")
    uploaded = 0

    for relative in UPLOAD_PATHS:
        local_target = data_path / relative
        if not local_target.exists():
            print(f"Skipping missing path: {local_target}")
            continue
        if local_target.is_file():
            blob_name = f"{gcs_prefix}/{relative}"
            _upload_file(client, bucket, local_target, blob_name)
            uploaded += 1
        else:
            uploaded += _upload_directory(
                client, bucket, local_target, f"{gcs_prefix}/{relative}"
            )

    gcs_uri = f"gs://{bucket}/{gcs_prefix}/"
    print(f"Uploaded {uploaded} objects to {gcs_uri}")
    return gcs_uri


def download_training_data(
    local_data_dir: str | Path,
    bucket: str = GCS_BUCKET,
    prefix: str = GCS_INPUT_PREFIX,
    project_id: str | None = None,
) -> Path:
    """Download training data from GCS to a local directory."""
    data_path = Path(local_data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    client = storage.Client(project=project_id)
    gcs_prefix = prefix.rstrip("/")

    blobs = client.list_blobs(bucket, prefix=f"{gcs_prefix}/")
    count = 0
    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        relative = blob.name[len(gcs_prefix) + 1 :]
        local_file = data_path / relative
        local_file.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_file))
        count += 1

    print(f"Downloaded {count} objects from gs://{bucket}/{gcs_prefix}/ to {data_path}")
    return data_path
