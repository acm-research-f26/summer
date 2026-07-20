#!/usr/bin/env python
# IMP3 runs/ -- full-scale entrypoint for the sub-agent credit trainer
# (agents/sub_agent_credit.py + orchestration/sub_agent_trainer.py): trains
# the specialists' shared adapter via the critic-free action/subgoal/switch
# GRPO credit split, orchestrator bypassed (fixed delegate-navigate ->
# delegate-interact -> stop schedule, same as Baseline A).
#
# Does NOT call sub_agent_trainer.train_sub_agents() directly -- that helper
# hardcodes SubAgentGRPOConfig()'s class default, which is keep_penalty=-3.5
# (HiPER's WebShop-tuned value, per PROJECT_GUIDE.md section 7.3/12b).
# This project's real ALFWorld reference (run_alfworld_lite.sh) uses -0.3.
# Fixed here by constructing SubAgentGRPOConfig explicitly with the
# ALFWorld-tuned value, per PROJECT_GUIDE.md's own priority-2 recommendation,
# rather than silently inheriting the wrong default for a full run.

import argparse
import json
import os
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RUNS_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _RUNS_DIR)
from common import new_run_id, run_dir, write_json  # noqa: E402

from agents.sub_agent_credit import SubAgentGRPOConfig  # noqa: E402
from orchestration.episode_runner import _ALFWORLD_CONFIG  # noqa: E402
from orchestration.seal.timing import RoundTimer  # noqa: E402
from orchestration.sub_agent_trainer import SubAgentTrainer  # noqa: E402

VARIANT = "sub_agent"
QUICK_NUM_GROUPS = 2  # matches sub_agent_trainer.train_sub_agents' own smoke-test default
QUICK_GROUP_SIZE = 2
ALFWORLD_KEEP_PENALTY = -0.3  # run_alfworld_lite.sh's real value; see module docstring above


def parse_args():
    p = argparse.ArgumentParser(description="Run the sub-agent credit trainer (specialists' shared adapter) on real ALFWorld tasks.")
    p.add_argument("--num-groups", type=int, default=10, help="Number of GRPO update rounds.")
    p.add_argument("--group-size", type=int, default=4, help="Rollouts of the same seeded task per group.")
    p.add_argument("--keep-penalty", type=float, default=ALFWORLD_KEEP_PENALTY, help="See module docstring: -0.3 is ALFWorld-tuned, -3.5 (the class default) is WebShop-tuned.")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--quick", action="store_true", help="Tiny smoke-test size instead of the full run (pipeline sanity check).")
    return p.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id or new_run_id()
    num_groups = QUICK_NUM_GROUPS if args.quick else args.num_groups
    group_size = QUICK_GROUP_SIZE if args.quick else args.group_size

    print(f"[{VARIANT}] run_id={run_id} num_groups={num_groups} group_size={group_size} keep_penalty={args.keep_penalty}")

    credit_cfg = SubAgentGRPOConfig(keep_penalty=args.keep_penalty)
    trainer = SubAgentTrainer(group_size=group_size, credit_cfg=credit_cfg)
    timer = RoundTimer()

    start = time.monotonic()
    history = []
    for group_id in range(num_groups):
        with timer.round(group_id):
            with timer.phase("episode_collection"):
                turns = trainer.collect_group(seed=group_id, config_path=_ALFWORLD_CONFIG)
            with timer.phase("update"):
                history.append(trainer.update(turns, group_id))
    elapsed = time.monotonic() - start

    out_dir = run_dir(VARIANT, run_id)
    write_json(os.path.join(out_dir, "result.json"), {
        "history": history,
        "_run_id": run_id,
        "_elapsed_s": elapsed,
        "_config": {"num_groups": num_groups, "group_size": group_size, "keep_penalty": args.keep_penalty},
    })
    timer.save(os.path.join(out_dir, "round_timings.jsonl"))

    print(f"[{VARIANT}] done in {elapsed:.1f}s -> {out_dir}")
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
