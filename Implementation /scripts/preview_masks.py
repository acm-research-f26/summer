#!/usr/bin/env python3
"""Render quick PNG previews of labeled segmentation masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fibroid_cavity.constants import CAVITY_LABEL, FIBROID_LABEL, NABOTHIAN_CYST_LABEL, UTERINE_WALL_LABEL
from fibroid_cavity.io import find_mask_paths, load_mask, patient_id_from_path
from fibroid_cavity.plotting import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


LABEL_NAMES = {
    0: "background",
    UTERINE_WALL_LABEL: "uterine wall",
    CAVITY_LABEL: "cavity",
    FIBROID_LABEL: "fibroid",
    NABOTHIAN_CYST_LABEL: "nabothian cyst",
}

LABEL_COLORS = {
    0: "#000000",
    UTERINE_WALL_LABEL: "#4e79a7",
    CAVITY_LABEL: "#f28e2b",
    FIBROID_LABEL: "#e15759",
    NABOTHIAN_CYST_LABEL: "#59a14f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-dir", type=Path, default=Path("data/UMD"), help="Directory containing NIfTI masks.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/mask_previews"), help="PNG output directory.")
    parser.add_argument("--limit", type=int, default=12, help="Maximum number of masks to preview.")
    parser.add_argument("--pattern", default="*", help="File search pattern passed to Path.rglob.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_paths = find_mask_paths(args.mask_dir, pattern=args.pattern)[: args.limit]

    if not mask_paths:
        raise SystemExit(f"No NIfTI masks found in {args.mask_dir}")

    for mask_path in mask_paths:
        mask, _ = load_mask(mask_path)
        patient_id = patient_id_from_path(mask_path)
        preview_path = args.output_dir / f"{patient_id}.png"
        _save_preview(mask, patient_id, preview_path)

    print(f"Wrote {len(mask_paths)} mask previews to {args.output_dir}")


def _save_preview(mask: np.ndarray, patient_id: str, output_path: Path) -> None:
    slice_index = _representative_slice(mask)
    slice_data = mask[:, :, slice_index].T

    cmap = ListedColormap([LABEL_COLORS[label] for label in sorted(LABEL_NAMES)])
    vmax = max(LABEL_NAMES)

    plt.figure(figsize=(6, 6))
    plt.imshow(slice_data, origin="lower", cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    plt.title(f"{patient_id} slice {slice_index}")
    plt.axis("off")

    present_labels = [label for label in sorted(LABEL_NAMES) if np.any(mask == label)]
    legend_handles = [
        Patch(facecolor=LABEL_COLORS[label], edgecolor="white", label=LABEL_NAMES[label])
        for label in present_labels
        if label != 0
    ]
    if legend_handles:
        plt.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def _representative_slice(mask: np.ndarray) -> int:
    fibroid_counts = (mask == FIBROID_LABEL).sum(axis=(0, 1))
    if fibroid_counts.max() > 0:
        return int(np.argmax(fibroid_counts))

    foreground_counts = (mask > 0).sum(axis=(0, 1))
    if foreground_counts.max() > 0:
        return int(np.argmax(foreground_counts))

    return mask.shape[2] // 2


if __name__ == "__main__":
    main()
