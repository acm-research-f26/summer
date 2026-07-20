#!/usr/bin/env python
# IMP3 runs/ -- full-scale entrypoint for Baseline A (orchestration/
# baseline_a_heuristic.py): the fixed delegate-navigate -> delegate-interact
# -> stop heuristic, no learning. This is the floor every trained variant
# needs to beat. Writes runs/baseline_a/results/<run_id>/result.json.

import argparse
import json
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RUNS_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _RUNS_DIR)
from common import new_run_id, read_json, run_dir, write_json  # noqa: E402

from orchestration.baseline_a_heuristic import run_baseline_a  # noqa: E402

VARIANT = "baseline_a"
QUICK_NUM_EPISODES = 8  # matches the module's own __main__ smoke-test size


def parse_args():
    p = argparse.ArgumentParser(description="Run Baseline A (fixed heuristic orchestrator) on real ALFWorld tasks.")
    p.add_argument("--num-episodes", type=int, default=30, help="Full-run default; the module's own smoke test used 8.")
    p.add_argument("--run-id", type=str, default=None, help="Shared across variants by run_all.sh; auto-generated if omitted.")
    p.add_argument("--quick", action="store_true", help="Tiny smoke-test size instead of the full run (pipeline sanity check).")
    return p.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id or new_run_id()
    num_episodes = QUICK_NUM_EPISODES if args.quick else args.num_episodes

    out_dir = run_dir(VARIANT, run_id)
    out_path = os.path.join(out_dir, "result.json")
    print(f"[{VARIANT}] run_id={run_id} num_episodes={num_episodes} -> {out_path}")

    start = time.monotonic()
    metrics = run_baseline_a(num_episodes=num_episodes, log_path=out_path)
    elapsed = time.monotonic() - start

    data = read_json(out_path)  # {"metrics": ..., "episodes": ...} already written by run_baseline_a
    data["_run_id"] = run_id
    data["_elapsed_s"] = elapsed
    data["_config"] = {"num_episodes": num_episodes}
    write_json(out_path, data)

    print(f"[{VARIANT}] done in {elapsed:.1f}s")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
