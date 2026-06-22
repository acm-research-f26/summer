"""Unified training orchestrator: dispatches to local GPU or Vertex AI."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

try:
    from .config import Config, load_config
    from .vertex_entrypoint import train
except ImportError:
    try:
        from config import Config, load_config
        from vertex_entrypoint import train
    except ImportError:
        from src.config import Config, load_config
        from src.vertex_entrypoint import train


def _cfg(config: Config | None) -> Config:
    return config or load_config()


def ensure_local_training_data(config: Config | None = None) -> Path:
    """Download manifest + image tiles from GCS if not already present locally.

    Returns the local training data directory path.
    """
    cfg = _cfg(config)
    local_data_dir = Path(cfg.paths.local_data_dir)
    manifest_path = local_data_dir / cfg.gcs.manifest_blob
    tiles_dir = local_data_dir / cfg.gcs.prefixes.image_tiles

    if manifest_path.exists() and tiles_dir.exists() and any(tiles_dir.iterdir()):
        return local_data_dir

    print(f"Training data not found at {local_data_dir}. Downloading from GCS...")
    try:
        try:
            from .gcs import download_training_data
        except ImportError:
            try:
                from gcs import download_training_data
            except ImportError:
                from src.gcs import download_training_data

        download_training_data(local_data_dir, config=cfg)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download training data to {local_data_dir}. "
            "Ensure GCP credentials are available and the bucket is accessible."
        ) from exc

    return local_data_dir


def resolve_local_artifacts_dir(config: Config | None = None, run_id: str | None = None) -> Path:
    """Return a timestamped artifacts subdirectory for the current run."""
    cfg = _cfg(config)
    resolved_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path(cfg.paths.artifacts_dir) / resolved_run_id


def run_local_training(
    config: Config | None = None,
    run_id: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    val_fraction: float | None = None,
    device: str | None = None,
) -> Path:
    """Ensure data is present locally, then run training and return artifacts path."""
    cfg = _cfg(config)
    training_dir = ensure_local_training_data(cfg)
    artifacts_dir = resolve_local_artifacts_dir(cfg, run_id)

    train(
        training_dir=training_dir,
        model_dir=artifacts_dir,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_fraction=val_fraction,
        device=device if device is not None else cfg.training.device,
        config=cfg,
    )

    print(f"\nLocal training complete. Artifacts saved to: {artifacts_dir.resolve()}")
    return artifacts_dir


def run_vertex_training(
    config: Config | None = None,
    run_id: str | None = None,
) -> str:
    """Submit a Vertex AI Custom Training job and return the run ID."""
    from google.cloud import aiplatform

    cfg = _cfg(config)
    vertex_cfg = cfg.vertex_ai
    train_cfg = cfg.training
    gcs_cfg = cfg.gcs

    resolved_run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    fuse_root = gcs_cfg.fuse_root
    output_prefix = gcs_cfg.prefixes.output_models
    model_fuse_dir = f"{fuse_root}/{output_prefix}/{resolved_run_id}/model"

    aiplatform.init(
        project=cfg.gcp.project_id,
        location=cfg.gcp.region,
        staging_bucket=vertex_cfg.staging_bucket,
    )

    display_name = f"{vertex_cfg.display_name_prefix}-{resolved_run_id}"
    job = aiplatform.CustomTrainingJob(
        display_name=display_name,
        script_path="src",
        container_uri=vertex_cfg.container_uri,
        requirements=list(vertex_cfg.requirements),
    )

    job.run(
        args=[
            "--epochs", str(train_cfg.epochs),
            "--batch-size", str(train_cfg.batch_size),
            "--training", fuse_root,
            "--model-dir", model_fuse_dir,
        ],
        replica_count=vertex_cfg.replica_count,
        machine_type=vertex_cfg.machine_type,
        accelerator_type=vertex_cfg.accelerator_type,
        accelerator_count=vertex_cfg.accelerator_count,
        base_output_dir=f"gs://{gcs_cfg.bucket_name}/{output_prefix}/{resolved_run_id}/",
        environment_variables={
            "SUMMER_CONFIG_PATH": f"{fuse_root}/config/default.yaml",
        },
        sync=False,
    )

    job_id = job._gca_resource.name.split("/")[-1] if job._gca_resource else "pending"
    print(f"RUN_ID={resolved_run_id}")
    print(f"JOB_ID={job_id}")
    print(f"DISPLAY_NAME={display_name}")
    print(f"CONTAINER={vertex_cfg.container_uri}")
    print(f"ARTIFACTS=gs://{gcs_cfg.bucket_name}/{output_prefix}/{resolved_run_id}/")
    print(
        f"CONSOLE=https://console.cloud.google.com/vertex-ai/training/{job_id}"
        f"?project={cfg.gcp.project_id}"
    )
    return resolved_run_id


def run_training(
    config: Config | None = None,
    runtime_override: str | None = None,
    run_id: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> Path | str:
    """Dispatch training to local or Vertex AI based on config (or override).

    Returns the artifacts path (local) or run ID (vertex).
    """
    cfg = _cfg(config)
    runtime = runtime_override or cfg.training.runtime

    if runtime == "local":
        return run_local_training(
            config=cfg,
            run_id=run_id,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
        )
    elif runtime == "vertex":
        return run_vertex_training(config=cfg, run_id=run_id)
    else:
        raise ValueError(
            f"Unknown runtime '{runtime}'. Must be 'local' or 'vertex'."
        )
