# IMP3 -- Baseline A: fixed heuristic orchestrator, no RL, the naive
# standard-practice floor. Reuses episode_runner.run_episode's
# `orchestrator_override` so this is the exact same execution path Baseline B
# and the real orchestrator use, just with a scripted action sequence instead
# of a learned/generated one.
#
# FIXED_ACTION_SEQUENCE was originally a single navigate-then-interact pair
# (["delegate-navigate", "delegate-interact", "stop"]) -- ported unchanged
# from implementation-2's WebShop version. Checked against real episode logs
# this session (runs/EXPERIMENT_LOG.md investigation, run_id 20260719_141730):
# every real ALFWorld task type needs at least 2 navigate+interact phases
# (go-to-source -> take -> go-to-destination -> put is the floor for even the
# simplest pick_and_place task), and heat/cool/clean-then-place task types
# need 3 (an extra go-to-appliance -> heat/cool/clean phase in between). One
# pair is structurally incapable of completing almost any real task,
# regardless of model quality -- this was a real bug, not a deliberate
# simplification, and it silently made Baseline A (and, worse,
# sub_agent_trainer.py/seal_trainer.py's *training* rollouts, which reuse
# this identical sequence) unable to ever observe a success to learn from.
#
# Fixed to 3 full navigate+interact cycles (covers every ALFWorld task type's
# structural need). Deliberately exactly 6 elements, no trailing "stop":
# episode_runner.run_episode's round_cap defaults to 6, and if an override
# sequence runs out before round_cap, execution silently falls through to the
# REAL (untrained) orchestrator's own decision for the remaining rounds --
# exactly the confound this fixed-schedule mechanism exists to avoid. An
# override exactly as long as round_cap never hits that fallthrough; the loop
# ends via round_cap naturally instead.
import json
from dataclasses import asdict
from typing import List

from orchestration.episode_runner import make_env, run_episode
from orchestration.interact_specialist import InteractSpecialist
from orchestration.navigate_specialist import NavigateSpecialist
from orchestration.orchestrator import Orchestrator, OrchestratorAction

FIXED_ACTION_SEQUENCE: List[OrchestratorAction] = [
    "delegate-navigate", "delegate-interact",
    "delegate-navigate", "delegate-interact",
    "delegate-navigate", "delegate-interact",
]


def run_baseline_a(num_episodes: int, log_path: str = None) -> dict:
    env = make_env()
    orchestrator = Orchestrator()  # unused for decisions, kept only so run_episode's signature matches Baseline B
    navigate = NavigateSpecialist()
    interact = InteractSpecialist()

    logs = []
    for _ in range(num_episodes):
        log = run_episode(env, orchestrator, navigate, interact, orchestrator_override=FIXED_ACTION_SEQUENCE)
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
