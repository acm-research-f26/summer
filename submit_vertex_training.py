#!/usr/bin/env python3
"""Submit a Vertex AI custom training job using project configuration.

Thin wrapper around src.training_runner.run_vertex_training. Prefer
run_training.py --runtime vertex for new workflows; this script is kept
for backward compatibility with existing docs and CI hooks.
"""

from __future__ import annotations

import argparse

from src.config import load_config
from src.training_runner import run_vertex_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU prebuilt container (no GPU quota required)",
    )
    args = parser.parse_args()

    config = load_config()

    if args.cpu:
        # Temporarily patch the container URI for the CPU-only run.
        # We rebuild config by overriding vertex_ai fields via a small shim.
        import dataclasses

        cpu_vertex_cfg = dataclasses.replace(
            config.vertex_ai,
            container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-xla.2-0.py310:latest",
            accelerator_count=0,
            accelerator_type="ACCELERATOR_TYPE_UNSPECIFIED",
        )
        config = dataclasses.replace(config, vertex_ai=cpu_vertex_cfg)

    run_vertex_training(config=config)


if __name__ == "__main__":
    main()
