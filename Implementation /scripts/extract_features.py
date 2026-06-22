#!/usr/bin/env python3
"""Extract fibroid-level features from labeled NIfTI masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fibroid_cavity.features import extract_patient_features
from fibroid_cavity.io import find_mask_paths, load_mask, patient_id_from_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-dir", type=Path, default=Path("data/UMD"), help="Directory containing NIfTI masks.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/fibroid_features.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--pattern", default="*", help="File search pattern passed to Path.rglob.")
    parser.add_argument(
        "--contact-iterations",
        type=int,
        default=1,
        help="Binary dilation iterations used when checking cavity boundary contact.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_paths = find_mask_paths(args.mask_dir, pattern=args.pattern)
    if not mask_paths:
        raise SystemExit(f"No NIfTI masks found in {args.mask_dir}")

    rows: list[dict[str, object]] = []
    failures: list[str] = []

    for mask_path in mask_paths:
        try:
            mask, spacing = load_mask(mask_path)
            patient_id = patient_id_from_path(mask_path)
            rows.extend(
                extract_patient_features(
                    mask=mask,
                    spacing=spacing,
                    patient_id=patient_id,
                    contact_iterations=args.contact_iterations,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep batch extraction moving.
            failures.append(f"{mask_path}: {exc}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)

    print(f"Wrote {len(rows)} fibroid rows to {args.output}")
    if failures:
        failure_path = args.output.with_suffix(".failures.txt")
        failure_path.write_text("\n".join(failures) + "\n", encoding="utf-8")
        print(f"Skipped {len(failures)} unreadable masks; details written to {failure_path}")


if __name__ == "__main__":
    main()
