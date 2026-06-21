"""Tabular preprocessing and ground-truth label generation for building records."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyproj

try:
    from config import Config, load_config
except ImportError:
    from src.config import Config, load_config


def run_tabular_preprocessing(
    csv_path: str | Path | None = None,
    output_path: str | Path | None = None,
    config: Config | None = None,
) -> pd.DataFrame:
    """Ingest buildings CSV, compute labels, transform coords, save manifest."""
    cfg = config or load_config()
    prep = cfg.preprocessing

    source_csv = csv_path or prep.paths.buildings_csv
    manifest_output = output_path or prep.paths.manifest_output

    df = pd.read_csv(source_csv)

    status_filter = prep.building_status_filter
    if status_filter:
        df = df[df["BuildingStatus"] == status_filter].copy()
        print(f"Filtered to BuildingStatus='{status_filter}': {len(df)} records")

    df["MaxGFA"] = df[list(prep.gfa_columns)].max(axis=1)
    df = df[df["MaxGFA"] > 0].copy()

    df["Est_MW"] = (df["MaxGFA"] * prep.mw.gfa_multiplier) / prep.mw.divisor
    conditions = [
        df["Est_MW"] < prep.mw.low_bound,
        (df["Est_MW"] >= prep.mw.low_bound) & (df["Est_MW"] <= prep.mw.medium_upper),
    ]
    df["target_label"] = np.select(
        conditions,
        [prep.tier.low, prep.tier.medium],
        default=prep.tier.large,
    ).astype(int)

    transformer = pyproj.Transformer.from_crs(
        prep.crs.source, prep.crs.target, always_xy=True
    )
    longitude, latitude = transformer.transform(df["X"].values, df["Y"].values)
    df["latitude"] = latitude
    df["longitude"] = longitude

    clean_df = df[list(prep.manifest_columns)].copy()
    output = Path(manifest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output, index=False)
    print(f"Tabular preprocessing complete. Manifest saved to {output}")
    return clean_df
