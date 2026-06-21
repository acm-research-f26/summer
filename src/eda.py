"""Exploratory data analysis helpers for tabular and satellite pipeline QA."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

IMPACT_TIER_LABELS = {
    0: "Low (Enterprise/Edge)",
    1: "Medium (Colocation)",
    2: "Large (Hyperscale)",
}


def load_building_context(
    manifest_path: str | Path,
    buildings_path: str | Path = "data/buildings.csv",
) -> pd.DataFrame:
    """Merge manifest coordinates and labels with building names for EDA."""
    manifest = pd.read_csv(manifest_path)
    buildings = pd.read_csv(buildings_path)[
        ["OBJECTID", "BuildingName", "BuildingStatus", "GFA", "BPGFA"]
    ]
    return manifest.merge(buildings, on="OBJECTID", how="left")


def verify_tile_alignment(
    manifest_path: str | Path,
    tile_dir: str | Path = "data/image_tiles",
    raw_dir: str | Path = "data/raw_satellite",
) -> pd.DataFrame:
    """Report missing or mismatched tiles and raw previews for each OBJECTID."""
    manifest = pd.read_csv(manifest_path)
    tile_path = Path(tile_dir)
    raw_path = Path(raw_dir)

    records: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        obj_id = int(row["OBJECTID"])
        npy_file = tile_path / f"tile_{obj_id}.npy"
        rgb_file = raw_path / f"tile_{obj_id}_rgb.png"
        records.append(
            {
                "OBJECTID": obj_id,
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "target_label": int(row["target_label"]),
                "has_npy": npy_file.exists(),
                "has_raw_rgb": rgb_file.exists(),
                "npy_shape": (
                    tuple(np.load(npy_file).shape) if npy_file.exists() else None
                ),
            }
        )

    report = pd.DataFrame(records)
    missing_npy = (~report["has_npy"]).sum()
    missing_raw = (~report["has_raw_rgb"]).sum()
    print(f"Manifest records: {len(report)}")
    print(f"Missing .npy tiles: {missing_npy}")
    print(f"Missing raw RGB previews: {missing_raw}")
    return report


def plot_tabular_eda(manifest_df: pd.DataFrame) -> Figure:
    """Plot label distribution, MaxGFA by tier, and geographic coverage."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    label_counts = manifest_df["target_label"].value_counts().sort_index()
    tier_names = [IMPACT_TIER_LABELS[int(label)] for label in label_counts.index]
    tier_colors = ["#4C78A8", "#F58518", "#E45756"]
    axes[0].bar(tier_names, label_counts.values, color=tier_colors)
    axes[0].set_title("Impact Tier Distribution")
    axes[0].set_ylabel("Building Count")
    axes[0].tick_params(axis="x", rotation=20)

    for label, color in zip([0, 1, 2], ["#4C78A8", "#F58518", "#E45756"], strict=True):
        subset = manifest_df[manifest_df["target_label"] == label]
        axes[1].hist(
            subset["MaxGFA"],
            bins=20,
            alpha=0.6,
            label=IMPACT_TIER_LABELS[label],
            color=color,
        )
    axes[1].set_title("MaxGFA by Impact Tier")
    axes[1].set_xlabel("Max Gross Floor Area (sq ft)")
    axes[1].set_ylabel("Count")
    axes[1].legend(fontsize=8)

    scatter = axes[2].scatter(
        manifest_df["longitude"],
        manifest_df["latitude"],
        c=manifest_df["target_label"],
        cmap="viridis",
        s=30,
        alpha=0.85,
    )
    axes[2].set_title("Building Locations (WGS84)")
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    fig.colorbar(scatter, ax=axes[2], label="Target Label")

    fig.tight_layout()
    return fig


def _approx_tile_bounds(
    latitude: float,
    longitude: float,
    half_extent_m: float = 640.0,
):
    """Approximate 1280 m tile footprint bounds in decimal degrees."""
    lat_offset = half_extent_m / 111_320.0
    lon_offset = half_extent_m / (111_320.0 * np.cos(np.radians(latitude)))
    return (
        longitude - lon_offset,
        latitude - lat_offset,
        longitude + lon_offset,
        latitude + lat_offset,
    )


def plot_geographic_tile_footprints(
    context_df: pd.DataFrame,
    sample_object_ids: list[int] | None = None,
    n_samples: int = 6,
) -> Figure:
    """Plot building coordinates and highlight sample tile footprints."""
    if sample_object_ids is None:
        sample_object_ids = context_df["OBJECTID"].head(n_samples).astype(int).tolist()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        context_df["longitude"],
        context_df["latitude"],
        c=context_df["target_label"],
        cmap="viridis",
        s=18,
        alpha=0.5,
        label="All buildings",
    )

    samples = context_df[context_df["OBJECTID"].isin(sample_object_ids)]
    for _, row in samples.iterrows():
        bounds = _approx_tile_bounds(float(row["latitude"]), float(row["longitude"]))
        rect = plt.Rectangle(
            (bounds[0], bounds[1]),
            bounds[2] - bounds[0],
            bounds[3] - bounds[1],
            fill=False,
            edgecolor="red",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.scatter(
            row["longitude"],
            row["latitude"],
            color="red",
            s=60,
            marker="x",
        )
        name = str(row.get("BuildingName", ""))[:24]
        ax.annotate(
            f"ID {int(row['OBJECTID'])}\n{name}",
            (row["longitude"], row["latitude"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color="black",
        )

    ax.set_title("Sample Tile Footprints (~1.28 km) Over Building Coordinates")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    return fig


def plot_satellite_samples(
    context_df: pd.DataFrame,
    raw_dir: str | Path = "data/raw_satellite",
    tile_dir: str | Path = "data/image_tiles",
    sample_object_ids: list[int] | None = None,
    n_samples: int = 4,
) -> Figure:
    """Visualize raw RGB previews alongside processed tensor band composites."""
    if sample_object_ids is None:
        sample_object_ids = context_df["OBJECTID"].head(n_samples).astype(int).tolist()

    raw_path = Path(raw_dir)
    tile_path = Path(tile_dir)
    n = len(sample_object_ids)
    fig, axes = plt.subplots(n, 4, figsize=(14, 3.5 * n))
    if n == 1:
        axes = np.array([axes])

    band_titles = ["Raw RGB Preview", "NPY RGB", "NPY NIR", "NPY SWIR"]
    for row_idx, obj_id in enumerate(sample_object_ids):
        row = context_df[context_df["OBJECTID"] == obj_id].iloc[0]
        rgb_preview = plt.imread(raw_path / f"tile_{obj_id}_rgb.png")
        tensor = np.load(tile_path / f"tile_{obj_id}.npy")

        images = [
            rgb_preview,
            np.transpose(tensor[:3], (1, 2, 0)),
            tensor[3],
            tensor[4],
        ]
        for col_idx, (image, title) in enumerate(zip(images, band_titles, strict=True)):
            ax = axes[row_idx, col_idx]
            if col_idx == 0:
                ax.imshow(image)
            elif col_idx == 1:
                ax.imshow(np.clip(image, 0.0, 1.0))
            else:
                ax.imshow(image, cmap="gray")
            if row_idx == 0:
                ax.set_title(title)
            ax.axis("off")

        building_name = str(row.get("BuildingName", "Unknown"))
        if col_idx == 0:
            ax.set_ylabel(
                f"ID {obj_id}\n{building_name[:22]}",
                fontsize=8,
                rotation=0,
                labelpad=42,
                va="center",
            )

    fig.suptitle("Satellite Tile QA: Raw Previews vs Processed Tensors", y=1.02)
    fig.tight_layout()
    return fig
