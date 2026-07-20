#!/usr/bin/env python
# IMP3 runs/ -- full-scale entrypoint for Baseline B (orchestration/
# baseline_b_rl_stop.py): GRPO-trained orchestrator stop policy, specialists
# frozen. Writes runs/baseline_b/results/<run_id>/result.json + round_timings.jsonl
# (the latter is what aggregate_and_log.py uses for SEAL's compute-overhead
# ratio in compare_variants.py).

import argparse
import json
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RUNS_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _RUNS_DIR)
from common import new_run_id, run_dir, write_json  # noqa: E402

from orchestration.baseline_b_rl_stop import DEFAULT_COST_COEF, train_baseline_b  # noqa: E402
from orchestration.seal.timing import RoundTimer  # noqa: E402

VARIANT = "baseline_b"
QUICK_NUM_GROUPS = 2  # matches the module's own __main__ smoke-test size
QUICK_GROUP_SIZE = 2


def parse_args():
    p = argparse.ArgumentParser(description="Run Baseline B (GRPO-trained orchestrator stop policy) on real ALFWorld tasks.")
    p.add_argument("--num-groups", type=int, default=10, help="Number of GRPO update rounds.")
    p.add_argument("--group-size", type=int, default=4, help="Rollouts per group (GRPO's group-relative baseline).")
    p.add_argument("--cost-coef", type=float, default=DEFAULT_COST_COEF, help="c in R = 10*success - c*rounds (still untuned per PROJECT_GUIDE.md).")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--quick", action="store_true", help="Tiny smoke-test size instead of the full run (pipeline sanity check).")
    return p.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id or new_run_id()
    num_groups = QUICK_NUM_GROUPS if args.quick else args.num_groups
    group_size = QUICK_GROUP_SIZE if args.quick else args.group_size

    print(f"[{VARIANT}] run_id={run_id} num_groups={num_groups} group_size={group_size} cost_coef={args.cost_coef}")

    timer = RoundTimer()
    start = time.monotonic()
    history = train_baseline_b(num_groups=num_groups, group_size=group_size, cost_coef=args.cost_coef, timer=timer)
    elapsed = time.monotonic() - start

    out_dir = run_dir(VARIANT, run_id)
    write_json(os.path.join(out_dir, "result.json"), {
        "history": history,
        "_run_id": run_id,
        "_elapsed_s": elapsed,
        "_config": {"num_groups": num_groups, "group_size": group_size, "cost_coef": args.cost_coef},
    })
    timer.save(os.path.join(out_dir, "round_timings.jsonl"))

    print(f"[{VARIANT}] done in {elapsed:.1f}s -> {out_dir}")
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
