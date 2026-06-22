#!/usr/bin/env python3
"""Create summary reports from an extracted fibroid feature CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fibroid_cavity.reporting import make_feature_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/fibroid_features.csv"),
        help="Input feature CSV path.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/eda"), help="Output report directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    make_feature_report(features, args.output_dir)
    print(f"Wrote feature report outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
