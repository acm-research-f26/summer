#!/usr/bin/env python
# IMP3 runs/ -- after all four variants for a given run_id have (attempted to)
# run, load whatever result.json files exist, compute a real SEAL-vs-Baseline-B
# verdict via orchestration.seal.compare_variants (never fabricated -- PENDING
# if either is missing), and append ONE dated section to
# implementation-3/EXPERIMENT_LOG.md summarizing all four variants. Safe to
# run even if some variants failed/were skipped: missing result files show up
# as "not run" rows, not crashes.

import argparse
import os
import sys
from datetime import datetime, timezone
from statistics import mean

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
from common import EXPERIMENT_LOG_PATH, RUNS_DIR, read_json  # noqa: E402

from orchestration.seal.compare_variants import compare  # noqa: E402

VARIANTS = ["baseline_a", "baseline_b", "sub_agent", "seal"]
VARIANT_LABELS = {
    "baseline_a": "Baseline A (fixed heuristic)",
    "baseline_b": "Baseline B (RL stop policy)",
    "sub_agent": "Sub-agent credit trainer",
    "seal": "SEAL",
}


def _result_path(variant: str, run_id: str) -> str:
    return os.path.join(RUNS_DIR, variant, "results", run_id, "result.json")


def _timings_mean_s(variant: str, run_id: str):
    path = os.path.join(RUNS_DIR, variant, "results", run_id, "round_timings.jsonl")
    if not os.path.exists(path):
        return None
    import json
    totals = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                totals.append(json.loads(line)["total_s"])
    return mean(totals) if totals else None


def _fmt(x, digits=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _row(variant: str, data) -> str:
    label = VARIANT_LABELS[variant]
    if data is None:
        return f"| {label} | — | — | not run | — |"

    elapsed = _fmt(data.get("_elapsed_s"), 0) + "s"

    if variant == "baseline_a":
        m = data["metrics"]
        return (f"| {label} | {_fmt(m['task_success_rate'])} | {_fmt(m['avg_delegation_rounds'])} "
                f"| cost_adj_return={_fmt(m['cost_adjusted_return_mean'])} | {elapsed} |")

    if variant == "baseline_b":
        last = data["history"][-1]
        return (f"| {label} | {_fmt(last.get('success_rate'))} | {_fmt(last.get('avg_delegation_rounds'))} "
                f"| reward_mean={_fmt(last.get('reward_mean'))}, pg_loss={_fmt(last.get('pg_loss_mean'))} | {elapsed} |")

    if variant == "sub_agent":
        last = data["history"][-1]
        kp = data["_config"]["keep_penalty"]
        return (f"| {label} | n/a (per-turn) | — "
                f"| pg_loss={_fmt(last.get('pg_loss_mean'))}, zero_reward_step_ratio={_fmt(last.get('zero_reward_step_ratio'))}, "
                f"keep_penalty={kp} | {elapsed} |")

    if variant == "seal":
        last = data["history"][-1]
        note = (last.get("applied_note") or "")[:80].replace("|", "/")
        return (f"| {label} | before={_fmt(last.get('before_success_rate'))} after={_fmt(last.get('after_success_rate_of_applied'))} | — "
                f"| applied_note=\"{note}\" | {elapsed} |")

    return f"| {label} | — | — | — | {elapsed} |"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    args = p.parse_args()
    run_id = args.run_id

    loaded = {}
    for variant in VARIANTS:
        path = _result_path(variant, run_id)
        if os.path.exists(path):
            loaded[variant] = read_json(path)
        else:
            loaded[variant] = None
            print(f"[aggregate] no result.json for {variant} (run_id={run_id}) -- marking 'not run'")

    baseline_a_path = _result_path("baseline_a", run_id) if loaded["baseline_a"] else "___missing___"
    baseline_b_history = loaded["baseline_b"]["history"] if loaded["baseline_b"] else None
    seal_history = loaded["seal"]["history"] if loaded["seal"] else None
    baseline_b_timing = _timings_mean_s("baseline_b", run_id)
    seal_timing = _timings_mean_s("seal", run_id)

    verdict = compare(
        baseline_a_path=baseline_a_path,
        baseline_b_history=baseline_b_history,
        seal_history=seal_history,
        baseline_b_timings_mean_s=baseline_b_timing,
        seal_timings_mean_s=seal_timing,
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [f"\n## Run `{run_id}` — {now}\n"]
    lines.append("| Variant | Success rate | Avg delegation rounds | Notes | Wall-clock |")
    lines.append("|---|---|---|---|---|")
    for variant in VARIANTS:
        lines.append(_row(variant, loaded[variant]))
    lines.append("")
    lines.append(f"**SEAL vs Baseline B verdict:** {verdict['verdict']}")
    lines.append("")
    lines.append(f"**Compute overhead ratio (SEAL / Baseline B, per-round wall-clock):** "
                 f"{_fmt(verdict['compute_overhead_ratio_seal_over_baseline_b'])}")
    lines.append("")
    result_files = ", ".join(f"`runs/{v}/results/{run_id}/`" for v in VARIANTS if loaded[v] is not None)
    lines.append(f"**Result files:** {result_files or '(none — all variants failed or were skipped)'}")
    lines.append("")

    with open(EXPERIMENT_LOG_PATH, "a") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[aggregate] appended run {run_id} to {EXPERIMENT_LOG_PATH}")


if __name__ == "__main__":
    main()
