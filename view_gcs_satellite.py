#!/usr/bin/env python3
"""List and display Sentinel-2 RGB previews stored in GCS after GEE extraction."""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from src.config import load_config
from src.gcs import (
    download_blob,
    list_tile_object_ids,
    metadata_blob_name,
    raw_blob_name,
)


def _parse_object_ids_from_prefix(
    bucket: str, prefix: str, project_id: str
) -> list[int]:
    """Fallback: derive OBJECTIDs from raw_satellite RGB blob names."""
    from google.cloud import storage

    client = storage.Client(project=project_id)
    pattern = re.compile(r"tile_(\d+)_rgb\.png$")
    ids: set[int] = set()
    for blob in client.list_blobs(bucket, prefix=f"{prefix}/"):
        match = pattern.search(blob.name)
        if match:
            ids.add(int(match.group(1)))
    return sorted(ids)


def _load_metadata(object_id: int, config) -> dict | None:
    blob = metadata_blob_name(object_id, config=config)
    try:
        payload = download_blob(blob, config=config)
    except Exception:
        return None
    return json.loads(payload.decode("utf-8"))


def _fetch_rgb(object_id: int, config):
    blob = raw_blob_name(object_id, "rgb", config=config)
    data = download_blob(blob, config=config)
    return plt.imread(io.BytesIO(data), format="png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print and display GCS satellite RGB previews from GEE extraction.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of tiles to show (default: all in GCS)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=5,
        help="Grid columns for the preview figure (default: 5)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Print inventory only; do not open a matplotlib window",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Save the preview grid to a file instead of showing interactively",
    )
    args = parser.parse_args(argv)

    config = load_config()
    bucket = config.gcs.bucket_name
    project_id = config.gcp.project_id
    raw_prefix = config.gcs.prefixes.raw_satellite
    tiles_prefix = config.gcs.prefixes.image_tiles

    object_ids = sorted(list_tile_object_ids(config=config))
    if not object_ids:
        object_ids = _parse_object_ids_from_prefix(bucket, raw_prefix, project_id)

    if not object_ids:
        print("No satellite tiles found in GCS.")
        print(f"  Bucket: gs://{bucket}/")
        print(f"  Prefixes: {tiles_prefix}/, {raw_prefix}/")
        return 1

    if args.limit is not None:
        object_ids = object_ids[: args.limit]

    print(f"GCS bucket: gs://{bucket}/")
    print(f"Found {len(object_ids)} tile(s) with imagery\n")
    print(f"{'OBJECTID':>8}  {'source':<22}  gs:// URI")
    print("-" * 72)

    records: list[tuple[int, object, str | None]] = []
    for obj_id in object_ids:
        meta = _load_metadata(obj_id, config)
        source = meta.get("source", "unknown") if meta else "no metadata"
        rgb_blob = raw_blob_name(obj_id, "rgb", config=config)
        uri = f"gs://{bucket}/{rgb_blob}"
        print(f"{obj_id:>8}  {source:<22}  {uri}")
        try:
            image = _fetch_rgb(obj_id, config)
            records.append((obj_id, image, source))
        except Exception as exc:
            print(f"         WARNING: could not load RGB preview — {exc}")

    if not records:
        print("\nNo RGB previews could be loaded.")
        return 1

    if args.no_display and args.save is None:
        print(
            f"\nLoaded {len(records)} RGB preview(s). "
            "Use --save or omit --no-display to plot."
        )
        return 0

    cols = max(1, args.cols)
    rows = math.ceil(len(records) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes_flat = (
        [axes]
        if rows == 1 and cols == 1
        else (axes.flatten() if hasattr(axes, "flatten") else list(axes))
    )

    for ax, (obj_id, image, source) in zip(axes_flat, records, strict=False):
        ax.imshow(image)
        ax.set_title(f"ID {obj_id}\n{source}", fontsize=8)
        ax.axis("off")

    for ax in axes_flat[len(records) :]:
        ax.axis("off")

    fig.suptitle(
        f"GCS satellite RGB previews ({len(records)} tiles)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=120, bbox_inches="tight")
        print(f"\nSaved preview grid to {args.save}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
