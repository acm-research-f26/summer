#!/usr/bin/env python3
"""CLI entry point for local or Vertex AI training.

Usage:
    python run_training.py                    # uses training.runtime from config
    python run_training.py --runtime local    # force local (Mac MPS/CPU)
    python run_training.py --runtime vertex   # submit Vertex AI job
    python run_training.py --device cpu       # override device for local runs
    python run_training.py --epochs 5         # override epoch count
"""

from __future__ import annotations

import argparse
import sys

from src.config import load_config
from src.training_runner import run_training


def main() -> None:
    config = load_config()

    parser = argparse.ArgumentParser(
        description="Train the data center vision classifier locally or on Vertex AI."
    )
    parser.add_argument(
        "--runtime",
        choices=["local", "vertex"],
        default=None,
        help="Where to run training (overrides config training.runtime)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default=None,
        help="Compute device for local training (overrides config training.device)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config training.epochs)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config training.batch_size)",
    )
    args = parser.parse_args()

    effective_runtime = args.runtime or config.training.runtime
    print(f"Runtime: {effective_runtime}")
    if effective_runtime == "local":
        effective_device = args.device or config.training.device
        print(f"Device:  {effective_device}")
    print(f"Epochs:  {args.epochs or config.training.epochs}")
    print()

    try:
        result = run_training(
            config=config,
            runtime_override=args.runtime,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
        )
        if isinstance(result, str):
            print(f"\nVertex AI job submitted. RUN_ID={result}")
        else:
            print(f"\nArtifacts written to: {result.resolve()}")
    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
