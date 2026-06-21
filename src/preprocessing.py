"""Tabular preprocessing and ground-truth label generation for building records."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyproj

GFA_COLUMNS = ["GFA", "BPGFA", "ApprovedGFA", "REATaxedGFA"]
MANIFEST_COLUMNS = ["OBJECTID", "latitude", "longitude", "target_label", "MaxGFA"]
TIER_LOW = 0
TIER_MEDIUM = 1
TIER_LARGE = 2
MW_LOW_BOUND = 20.0
MW_MEDIUM_UPPER = 90.0


def run_tabular_preprocessing(
    csv_path: str | Path,
    output_path: str | Path = "data/parsed_manifest.csv",
) -> pd.DataFrame:
    """Ingest buildings CSV, compute labels, transform coords, save manifest."""
    df = pd.read_csv(csv_path)

    df["MaxGFA"] = df[GFA_COLUMNS].max(axis=1)
    df = df[df["MaxGFA"] > 0].copy()

    df["Est_MW"] = (df["MaxGFA"] * 150) / 1_000_000
    conditions = [
        df["Est_MW"] < MW_LOW_BOUND,
        (df["Est_MW"] >= MW_LOW_BOUND) & (df["Est_MW"] <= MW_MEDIUM_UPPER),
    ]
    df["target_label"] = np.select(
        conditions,
        [TIER_LOW, TIER_MEDIUM],
        default=TIER_LARGE,
    ).astype(int)

    transformer = pyproj.Transformer.from_crs("EPSG:2283", "EPSG:4326", always_xy=True)
    longitude, latitude = transformer.transform(df["X"].values, df["Y"].values)
    df["latitude"] = latitude
    df["longitude"] = longitude

    clean_df = df[MANIFEST_COLUMNS].copy()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output, index=False)
    print(f"Tabular preprocessing complete. Manifest saved to {output}")
    return clean_df
