#!/usr/bin/env python
# IMP3 runs/ -- full-scale entrypoint for the SEAL self-editing variant
# (orchestration/seal/seal_trainer.py). Reimplements seal_trainer.train_seal's
# loop locally (rather than calling it) only so its two side-effect files
# (round timings, forgetting curve) land under this run's own results
# directory instead of train_seal()'s hardcoded relative paths.

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RUNS_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _RUNS_DIR)
from common import new_run_id, run_dir, write_json  # noqa: E402

from orchestration.episode_runner import _ALFWORLD_CONFIG  # noqa: E402
from orchestration.seal.seal_trainer import SealTrainer  # noqa: E402

VARIANT = "seal"
# matches seal_trainer's own __main__ smoke-test sizes
QUICK_NUM_ROUNDS = 1
QUICK_BATCH_SIZE = 2
QUICK_HELD_OUT = 2


def parse_args():
    p = argparse.ArgumentParser(description="Run the SEAL self-editing variant on real ALFWorld tasks.")
    p.add_argument("--num-rounds", type=int, default=5, help="Outer note-writer RL rounds.")
    p.add_argument("--train-batch-size", type=int, default=3, help="Episodes collected per round to summarize for note-writing.")
    p.add_argument("--eval-batch-size", type=int, default=3, help="Episodes used for each before/after success-rate measurement.")
    p.add_argument("--num-candidates", type=int, default=2, help="Candidate notes sampled per round (K in the outer GRPO).")
    p.add_argument("--held-out-size", type=int, default=3, help="Fixed held-out task count for the forgetting tracker.")
    p.add_argument("--forgetting-check-every", type=int, default=2, help="Re-evaluate the held-out set every N rounds.")
    p.add_argument("--note-lr", type=float, default=1e-5)
    p.add_argument("--self-edit-lr", type=float, default=5e-5)
    p.add_argument("--self-edit-steps", type=int, default=3)
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--quick", action="store_true", help="Tiny smoke-test size instead of the full run (pipeline sanity check).")
    return p.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id or new_run_id()
    num_rounds = QUICK_NUM_ROUNDS if args.quick else args.num_rounds
    train_bs = QUICK_BATCH_SIZE if args.quick else args.train_batch_size
    eval_bs = QUICK_BATCH_SIZE if args.quick else args.eval_batch_size
    num_candidates = QUICK_BATCH_SIZE if args.quick else args.num_candidates
    held_out = QUICK_HELD_OUT if args.quick else args.held_out_size

    print(f"[{VARIANT}] run_id={run_id} num_rounds={num_rounds} train_bs={train_bs} eval_bs={eval_bs} "
          f"num_candidates={num_candidates} held_out={held_out}")

    trainer = SealTrainer(
        config_path=_ALFWORLD_CONFIG,
        train_batch_size=train_bs,
        eval_batch_size=eval_bs,
        num_candidates=num_candidates,
        held_out_size=held_out,
        forgetting_check_every=args.forgetting_check_every,
        note_lr=args.note_lr,
        self_edit_lr=args.self_edit_lr,
        self_edit_steps=args.self_edit_steps,
    )

    start = time.monotonic()
    history = []
    for round_idx in range(num_rounds):
        result = trainer.run_round(round_idx, train_seed_base=round_idx * 100, eval_seed_base=round_idx * 100 + 50)
        history.append(asdict(result))
    elapsed = time.monotonic() - start

    out_dir = run_dir(VARIANT, run_id)
    write_json(os.path.join(out_dir, "result.json"), {
        "history": history,
        "_run_id": run_id,
        "_elapsed_s": elapsed,
        "_config": {
            "num_rounds": num_rounds, "train_batch_size": train_bs, "eval_batch_size": eval_bs,
            "num_candidates": num_candidates, "held_out_size": held_out,
            "forgetting_check_every": args.forgetting_check_every,
            "note_lr": args.note_lr, "self_edit_lr": args.self_edit_lr, "self_edit_steps": args.self_edit_steps,
        },
    })
    trainer.timer.save(os.path.join(out_dir, "round_timings.jsonl"))
    trainer.forgetting_tracker.save(os.path.join(out_dir, "forgetting_curve.json"))

    print(f"[{VARIANT}] done in {elapsed:.1f}s -> {out_dir}")
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
