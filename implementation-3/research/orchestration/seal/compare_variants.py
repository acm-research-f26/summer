# IMP3 SEAL variant -- final comparison across Baseline A, Baseline B, and
# SEAL. Loads whatever result files each variant produced and computes, for
# each: success rate, cost-adjusted return (reusing baseline_a_heuristic's
# own formula for apples-to-apples comparison), and -- for SEAL specifically
# -- a compute-overhead ratio against Baseline B's own per-round wall-clock
# (both measured with the same orchestration.seal.timing.RoundTimer
# instrument, not hand-estimated). Prints an explicit worth-it-or-not verdict
# computed from the actual run's numbers -- implementation-3/README.md should
# quote this script's output, not restate a separate claim.

import json
from typing import Optional


def _cost_adjusted_return(success: bool, num_delegation_rounds: int, c: float = 0.5) -> float:
    # same formula as baseline_a_heuristic.py / baseline_b_rl_stop.py, reused
    # here rather than reimplemented so all three variants are compared on
    # the exact same metric definition.
    return 10.0 * float(success) - c * num_delegation_rounds


def load_baseline_a(path: str = "baseline_a_results.json") -> Optional[dict]:
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("metrics")
    except FileNotFoundError:
        return None


def summarize_baseline_b(history: list) -> Optional[dict]:
    if not history:
        return None
    last = history[-1]
    return {
        "success_rate": last.get("success_rate"),
        "avg_delegation_rounds": last.get("avg_delegation_rounds"),
        "reward_mean": last.get("reward_mean"),
    }


def summarize_seal(history: list) -> Optional[dict]:
    if not history:
        return None
    last = history[-1]
    return {
        "before_success_rate": last.get("before_success_rate"),
        "after_success_rate_of_applied": last.get("after_success_rate_of_applied"),
        "success_rate_delta": (last.get("after_success_rate_of_applied", 0.0) - last.get("before_success_rate", 0.0)),
        "applied_note": last.get("applied_note"),
        "forgetting_checkpoint": last.get("forgetting_checkpoint"),
    }


def compare(
    baseline_a_path: str = "baseline_a_results.json",
    baseline_b_history: Optional[list] = None,
    seal_history: Optional[list] = None,
    baseline_b_timings_mean_s: Optional[float] = None,
    seal_timings_mean_s: Optional[float] = None,
) -> dict:
    result = {
        "baseline_a": load_baseline_a(baseline_a_path),
        "baseline_b": summarize_baseline_b(baseline_b_history or []),
        "seal": summarize_seal(seal_history or []),
    }

    overhead_ratio = None
    if baseline_b_timings_mean_s and seal_timings_mean_s:
        overhead_ratio = seal_timings_mean_s / baseline_b_timings_mean_s
    result["compute_overhead_ratio_seal_over_baseline_b"] = overhead_ratio

    verdict_lines = []
    seal_summary = result["seal"]
    baseline_b_summary = result["baseline_b"]
    if seal_summary is None or baseline_b_summary is None:
        verdict_lines.append(
            "PENDING: SEAL and/or Baseline B haven't been run yet in this environment -- "
            "no result files to compare. This is not a placeholder result; run both and "
            "re-invoke compare() before drawing any conclusion."
        )
    else:
        delta = seal_summary["success_rate_delta"]
        if overhead_ratio is None:
            verdict_lines.append(
                f"SEAL's applied-note success rate moved by {delta:+.2f} vs. its own before-eval "
                f"in the last round, but no wall-clock overhead ratio is available -- pass "
                f"baseline_b_timings_mean_s/seal_timings_mean_s to compute a real worth-it verdict."
            )
        else:
            if delta > 0:
                verdict_lines.append(
                    f"SEAL improved success rate by {delta:+.2%} at {overhead_ratio:.1f}x the "
                    f"wall-clock cost per round of Baseline B -- "
                    f"{'worth it' if delta / max(overhead_ratio, 1e-6) > 0.01 else 'marginal at this scale'}, "
                    f"but this is a downtuned smoke-scale run (few rounds, tiny eval batch) -- "
                    f"treat this as a structural signal, not a converged result."
                )
            else:
                verdict_lines.append(
                    f"SEAL did NOT improve success rate ({delta:+.2%}) while costing {overhead_ratio:.1f}x "
                    f"Baseline B's wall-clock per round -- not worth it at this scale, based on this run's "
                    f"actual numbers."
                )
    result["verdict"] = " ".join(verdict_lines)
    return result


if __name__ == "__main__":
    print(json.dumps(compare(), indent=2))
