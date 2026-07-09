#!/usr/bin/env python3
"""
Offline evaluation: compare baseline click paths vs ARP-guided paths.

Runs on sample DOM fixtures (no browser or API keys needed). Produces
results/comparison.json with aggregate metrics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running as script from implementation_2/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.sample_pages import SIMULATED_TRAJECTORIES, task_succeeded
from src.action_relevance import ActionRelevanceScorer
from src.trajectory_logger import TrajectoryLogger


def simulate_run(
    task_id: str,
    task: str,
    elements: list[dict],
    click_sequence: list[int],
    mode: str,
    scorer: ActionRelevanceScorer,
    logger: TrajectoryLogger,
) -> None:
    logger.start_run(task_id, mode, task)
    ranked = scorer.rank(task, elements)

    for idx in click_sequence:
        el = next((e for e in elements if e["index"] == idx), None)
        target = el["text"] if el else f"element_{idx}"
        in_top_k = scorer.is_in_top_k(idx, ranked)
        score = scorer.score_for_index(idx, ranked)
        logger.log_step("click", target, relevance_score=score, in_top_k=in_top_k)


def aggregate(records: list) -> dict:
    if not records:
        return {}

    n = len(records)
    return {
        "runs": n,
        "success_rate": sum(1 for r in records if r.success) / n,
        "avg_steps": sum(r.steps for r in records) / n,
        "avg_wall_time_sec": sum(r.wall_time_sec for r in records) / n,
        "avg_irrelevant_clicks": sum(r.irrelevant_clicks for r in records) / n,
    }


def main() -> None:
    results_dir = ROOT / "results"
    runs_dir = results_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    scorer = ActionRelevanceScorer(top_k=3)

    for task_id, spec in SIMULATED_TRAJECTORIES.items():
        logger_b = TrajectoryLogger(runs_dir)
        simulate_run(
            task_id, spec["task"], spec["elements"],
            spec["baseline_clicks"], "baseline", scorer, logger_b,
        )
        logger_b.finish_run(task_succeeded(spec["baseline_clicks"], spec["goal_element"]))

        logger_a = TrajectoryLogger(runs_dir)
        simulate_run(
            task_id, spec["task"], spec["elements"],
            spec["arp_clicks"], "arp", scorer, logger_a,
        )
        logger_a.finish_run(task_succeeded(spec["arp_clicks"], spec["goal_element"]))

    # Re-read from saved files for clean aggregation
    baseline_data = []
    arp_data = []
    for task_id in SIMULATED_TRAJECTORIES:
        with open(runs_dir / f"{task_id}_baseline.json") as f:
            baseline_data.append(json.load(f))
        with open(runs_dir / f"{task_id}_arp.json") as f:
            arp_data.append(json.load(f))

    def agg_from_dicts(data: list[dict]) -> dict:
        n = len(data)
        return {
            "runs": n,
            "success_rate": sum(1 for d in data if d["success"]) / n,
            "avg_steps": sum(d["steps"] for d in data) / n,
            "avg_wall_time_sec": sum(d["wall_time_sec"] for d in data) / n,
            "avg_irrelevant_clicks": sum(d["irrelevant_clicks"] for d in data) / n,
        }

    baseline_agg = agg_from_dicts(baseline_data)
    arp_agg = agg_from_dicts(arp_data)

    def pct_change(baseline_val: float, arp_val: float) -> float:
        if baseline_val == 0:
            return 0.0
        return round(100 * (baseline_val - arp_val) / baseline_val, 1)

    comparison = {
        "eval_type": "offline_sample_pages",
        "tasks": list(SIMULATED_TRAJECTORIES.keys()),
        "baseline": baseline_agg,
        "arp": arp_agg,
        "improvement": {
            "steps_reduction_pct": pct_change(baseline_agg["avg_steps"], arp_agg["avg_steps"]),
            "irrelevant_clicks_reduction_pct": pct_change(
                baseline_agg["avg_irrelevant_clicks"], arp_agg["avg_irrelevant_clicks"]
            ),
            "success_rate_delta": round(arp_agg["success_rate"] - baseline_agg["success_rate"], 3),
        },
        "per_task": [],
    }

    for b, a in zip(baseline_data, arp_data):
        comparison["per_task"].append({
            "task_id": b["task_id"],
            "baseline_steps": b["steps"],
            "arp_steps": a["steps"],
            "baseline_irrelevant": b["irrelevant_clicks"],
            "arp_irrelevant": a["irrelevant_clicks"],
        })

    out_path = results_dir / "comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)

    print("=== Implementation 2 — Offline ARP Evaluation ===\n")
    print(f"Tasks: {', '.join(comparison['tasks'])}")
    print(f"\nBaseline  — avg steps: {baseline_agg['avg_steps']:.1f}, irrelevant clicks: {baseline_agg['avg_irrelevant_clicks']:.1f}")
    print(f"ARP       — avg steps: {arp_agg['avg_steps']:.1f}, irrelevant clicks: {arp_agg['avg_irrelevant_clicks']:.1f}")
    print(f"\nSteps reduction: {comparison['improvement']['steps_reduction_pct']}%")
    print(f"Irrelevant click reduction: {comparison['improvement']['irrelevant_clicks_reduction_pct']}%")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
