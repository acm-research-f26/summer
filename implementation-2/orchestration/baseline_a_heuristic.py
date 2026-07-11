# IMP2 Part 3 -- Baseline A: fixed heuristic orchestrator, no RL, the naive
# standard-practice floor. Stopping rule is explicit and fixed: delegate to
# search once, then delegate to compare once, then stop -- reusing
# episode_runner.run_episode's `orchestrator_override` so this is the exact
# same execution path Baseline B and the real orchestrator use, just with a
# scripted action sequence instead of a learned/generated one.

import json
from dataclasses import asdict
from typing import List

from orchestration.compare_specialist import CompareSpecialist
from orchestration.episode_runner import make_env, run_episode
from orchestration.orchestrator import Orchestrator, OrchestratorAction
from orchestration.search_specialist import SearchSpecialist

FIXED_ACTION_SEQUENCE: List[OrchestratorAction] = ["delegate-search", "delegate-compare", "stop"]


def run_baseline_a(num_episodes: int, log_path: str = None) -> dict:
    env = make_env()
    orchestrator = Orchestrator()  # unused for decisions, kept only so run_episode's signature matches Baseline B
    search = SearchSpecialist()
    compare = CompareSpecialist()

    logs = []
    for _ in range(num_episodes):
        log = run_episode(env, orchestrator, search, compare, orchestrator_override=FIXED_ACTION_SEQUENCE)
        logs.append(log)

    successes = [l.success for l in logs]
    rounds_used = [l.num_delegation_rounds for l in logs]
    metrics = {
        "task_success_rate": sum(successes) / len(successes),
        "avg_delegation_rounds": sum(rounds_used) / len(rounds_used),
        "cost_adjusted_return_mean": sum(_cost_adjusted_return(l) for l in logs) / len(logs),
        "cost_adjusted_return_std": _std([_cost_adjusted_return(l) for l in logs]),
        "num_episodes": num_episodes,
    }

    if log_path:
        with open(log_path, "w") as f:
            json.dump({"metrics": metrics, "episodes": [l.to_dict() for l in logs]}, f, indent=2)

    return metrics


def _cost_adjusted_return(log, c: float = 0.5) -> float:
    return 10.0 * float(log.success) - c * log.num_delegation_rounds


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5


if __name__ == "__main__":
    metrics = run_baseline_a(num_episodes=8, log_path="baseline_a_results.json")
    print(json.dumps(metrics, indent=2))
