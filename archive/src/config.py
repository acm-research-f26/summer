"""Centralized configuration loading for the data center vision pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_BUNDLED_CONFIG = Path(__file__).resolve().parent / "configs" / "default.yaml"
_BUNDLED_FLAT_CONFIG = Path(__file__).resolve().parent / "default.yaml"
_REPO_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
DEFAULT_CONFIG_PATH = next(
    (path for path in (_BUNDLED_CONFIG, _BUNDLED_FLAT_CONFIG, _REPO_CONFIG) if path.exists()),
    _REPO_CONFIG,
)
CONFIG_ENV_VAR = "SUMMER_CONFIG_PATH"


@dataclass(frozen=True)
class GCPConfig:
    project_id: str
    region: str


@dataclass(frozen=True)
class GCSPrefixes:
    image_tiles: str
    raw_satellite: str
    output_models: str


@dataclass(frozen=True)
class GCSConfig:
    bucket_name: str
    fuse_root: str
    manifest_blob: str
    prefixes: GCSPrefixes


@dataclass(frozen=True)
class EarthEngineConfig:
    s2_collection: str
    band_names: tuple[str, ...]
    start_date: str
    end_date: str
    max_cloud_cover: int
    scale_m: int
    reproject_crs: str
    reflectance_scale: float
    source_id: str
    mock_source_id: str
    progress_log_every: int


@dataclass(frozen=True)
class DatasetConfig:
    tile_shape: tuple[int, int, int]
    band_names: tuple[str, ...]
    tile_px: int


@dataclass(frozen=True)
class TierConfig:
    low: int
    medium: int
    large: int


@dataclass(frozen=True)
class MWConfig:
    low_bound: float
    medium_upper: float
    gfa_multiplier: int
    divisor: int


@dataclass(frozen=True)
class CRSConfig:
    source: str
    target: str


@dataclass(frozen=True)
class PreprocessingPaths:
    buildings_csv: str
    manifest_output: str


@dataclass(frozen=True)
class PreprocessingConfig:
    building_status_filter: str | None
    gfa_columns: tuple[str, ...]
    manifest_columns: tuple[str, ...]
    tier: TierConfig
    mw: MWConfig
    crs: CRSConfig
    paths: PreprocessingPaths


@dataclass(frozen=True)
class ModelConfig:
    num_classes: int
    backbone: str
    in_channels: int


@dataclass(frozen=True)
class TrainingConfig:
    runtime: str  # "local" | "vertex"
    device: str   # "auto" | "mps" | "cuda" | "cpu"
    epochs: int
    batch_size: int
    learning_rate: float
    val_fraction: float
    random_state: int
    plot_dpi: int


@dataclass(frozen=True)
class VertexPipelineConfig:
    epochs: int
    batch_size: int


@dataclass(frozen=True)
class VertexAIConfig:
    container_uri: str
    staging_bucket: str
    machine_type: str
    accelerator_type: str
    accelerator_count: int
    replica_count: int
    display_name_prefix: str
    requirements: tuple[str, ...]
    default_training_dir: str
    default_model_dir: str
    pipeline: VertexPipelineConfig


@dataclass(frozen=True)
class LoggingConfig:
    logger_name: str
    log_path: str
    level: str
    date_format: str
    message_format: str


@dataclass(frozen=True)
class PathsConfig:
    local_data_dir: str
    local_tiles_dir: str
    local_raw_dir: str
    artifacts_dir: str


@dataclass(frozen=True)
class PipelineConfig:
    use_mock: bool
    dev_limit: int | None
    clear_satellite_gcs_before_extract: bool


@dataclass(frozen=True)
class EDAConfig:
    impact_tier_labels: dict[int, str]
    tier_colors: tuple[str, ...]
    geographic_sample_count: int
    satellite_sample_count: int


@dataclass(frozen=True)
class Config:
    gcp: GCPConfig
    gcs: GCSConfig
    earth_engine: EarthEngineConfig
    dataset: DatasetConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    training: TrainingConfig
    vertex_ai: VertexAIConfig
    logging: LoggingConfig
    paths: PathsConfig
    pipeline: PipelineConfig
    eda: EDAConfig

    @property
    def tile_half_extent_m(self) -> float:
        return (self.dataset.tile_px * self.earth_engine.scale_m) / 2

    @property
    def tile_footprint_m(self) -> int:
        return self.dataset.tile_px * self.earth_engine.scale_m


def _build_config(raw: dict[str, Any]) -> Config:
    gcs_raw = raw["gcs"]
    ee_raw = raw["earth_engine"]
    dataset_raw = raw["dataset"]
    prep_raw = raw["preprocessing"]
    vertex_raw = raw["vertex_ai"]
    eda_raw = raw["eda"]

    tile_px = int(dataset_raw["tile_px"])
    scale_m = int(ee_raw["scale_m"])

    return Config(
        gcp=GCPConfig(
            project_id=raw["gcp"]["project_id"],
            region=raw["gcp"]["region"],
        ),
        gcs=GCSConfig(
            bucket_name=gcs_raw["bucket_name"],
            fuse_root=gcs_raw["fuse_root"],
            manifest_blob=gcs_raw["manifest_blob"],
            prefixes=GCSPrefixes(
                image_tiles=gcs_raw["prefixes"]["image_tiles"],
                raw_satellite=gcs_raw["prefixes"]["raw_satellite"],
                output_models=gcs_raw["prefixes"]["output_models"],
            ),
        ),
        earth_engine=EarthEngineConfig(
            s2_collection=ee_raw["s2_collection"],
            band_names=tuple(ee_raw["band_names"]),
            start_date=ee_raw["start_date"],
            end_date=ee_raw["end_date"],
            max_cloud_cover=int(ee_raw["max_cloud_cover"]),
            scale_m=scale_m,
            reproject_crs=ee_raw["reproject_crs"],
            reflectance_scale=float(ee_raw["reflectance_scale"]),
            source_id=ee_raw["source_id"],
            mock_source_id=ee_raw["mock_source_id"],
            progress_log_every=int(ee_raw["progress_log_every"]),
        ),
        dataset=DatasetConfig(
            tile_shape=tuple(int(v) for v in dataset_raw["tile_shape"]),
            band_names=tuple(dataset_raw["band_names"]),
            tile_px=tile_px,
        ),
        preprocessing=PreprocessingConfig(
            building_status_filter=prep_raw.get("building_status_filter") or None,
            gfa_columns=tuple(prep_raw["gfa_columns"]),
            manifest_columns=tuple(prep_raw["manifest_columns"]),
            tier=TierConfig(
                low=int(prep_raw["tier"]["low"]),
                medium=int(prep_raw["tier"]["medium"]),
                large=int(prep_raw["tier"]["large"]),
            ),
            mw=MWConfig(
                low_bound=float(prep_raw["mw"]["low_bound"]),
                medium_upper=float(prep_raw["mw"]["medium_upper"]),
                gfa_multiplier=int(prep_raw["mw"]["gfa_multiplier"]),
                divisor=int(prep_raw["mw"]["divisor"]),
            ),
            crs=CRSConfig(
                source=prep_raw["crs"]["source"],
                target=prep_raw["crs"]["target"],
            ),
            paths=PreprocessingPaths(
                buildings_csv=prep_raw["paths"]["buildings_csv"],
                manifest_output=prep_raw["paths"]["manifest_output"],
            ),
        ),
        model=ModelConfig(
            num_classes=int(raw["model"]["num_classes"]),
            backbone=raw["model"]["backbone"],
            in_channels=int(raw["model"]["in_channels"]),
        ),
        training=TrainingConfig(
            runtime=str(raw["training"].get("runtime", "local")),
            device=str(raw["training"].get("device", "auto")),
            epochs=int(raw["training"]["epochs"]),
            batch_size=int(raw["training"]["batch_size"]),
            learning_rate=float(raw["training"]["learning_rate"]),
            val_fraction=float(raw["training"]["val_fraction"]),
            random_state=int(raw["training"]["random_state"]),
            plot_dpi=int(raw["training"]["plot_dpi"]),
        ),
        vertex_ai=VertexAIConfig(
            container_uri=vertex_raw["container_uri"],
            staging_bucket=vertex_raw["staging_bucket"],
            machine_type=vertex_raw["machine_type"],
            accelerator_type=vertex_raw["accelerator_type"],
            accelerator_count=int(vertex_raw["accelerator_count"]),
            replica_count=int(vertex_raw["replica_count"]),
            display_name_prefix=vertex_raw["display_name_prefix"],
            requirements=tuple(vertex_raw["requirements"]),
            default_training_dir=vertex_raw["default_training_dir"],
            default_model_dir=vertex_raw["default_model_dir"],
            pipeline=VertexPipelineConfig(
                epochs=int(vertex_raw["pipeline"]["epochs"]),
                batch_size=int(vertex_raw["pipeline"]["batch_size"]),
            ),
        ),
        logging=LoggingConfig(
            logger_name=raw["logging"]["logger_name"],
            log_path=raw["logging"]["log_path"],
            level=raw["logging"]["level"],
            date_format=raw["logging"]["date_format"],
            message_format=raw["logging"]["message_format"],
        ),
        paths=PathsConfig(
            local_data_dir=raw["paths"]["local_data_dir"],
            local_tiles_dir=raw["paths"]["local_tiles_dir"],
            local_raw_dir=raw["paths"]["local_raw_dir"],
            artifacts_dir=raw["paths"]["artifacts_dir"],
        ),
        pipeline=PipelineConfig(
            use_mock=bool(raw["pipeline"]["use_mock"]),
            dev_limit=raw["pipeline"]["dev_limit"],
            clear_satellite_gcs_before_extract=bool(
                raw["pipeline"]["clear_satellite_gcs_before_extract"]
            ),
        ),
        eda=EDAConfig(
            impact_tier_labels={
                int(k): v for k, v in eda_raw["impact_tier_labels"].items()
            },
            tier_colors=tuple(eda_raw["tier_colors"]),
            geographic_sample_count=int(eda_raw["geographic_sample_count"]),
            satellite_sample_count=int(eda_raw["satellite_sample_count"]),
        ),
    )


@lru_cache(maxsize=4)
def load_config(path: str | Path | None = None) -> Config:
    """Load and parse the project configuration YAML file."""
    resolved = (
        path
        if path is not None
        else os.environ.get(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH)
    )
    config_path = Path(resolved)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        msg = f"Configuration file must contain a mapping: {config_path}"
        raise ValueError(msg)
    return _build_config(raw)
