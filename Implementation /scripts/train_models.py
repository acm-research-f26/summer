#!/usr/bin/env python3
"""Train leakage-aware cavity-contact classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fibroid_cavity.constants import PREDICTOR_COLUMNS
from fibroid_cavity.modeling import evaluate_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/fibroid_features.csv"),
        help="Input feature CSV path.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports"), help="Directory for reports and models.")
    parser.add_argument("--splits", type=int, default=5, help="Maximum number of grouped CV folds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--predictors",
        nargs="+",
        default=PREDICTOR_COLUMNS,
        help="Predictor columns to use. Defaults to the leakage-aware feature set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features = pd.read_csv(args.features)
    metrics = evaluate_models(
        features=features,
        output_dir=args.output_dir,
        predictors=args.predictors,
        n_splits=args.splits,
        random_state=args.seed,
    )
    print(metrics.groupby("model")["auc"].mean().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
