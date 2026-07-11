# IMP2 Part 2 -- wires the orchestrator + two specialists together against a
# live WebShop env for one full episode: orchestrator decides, delegates,
# sub-agent executes (Part 1's mechanism), result returns to orchestrator,
# repeat until stop or a round cap. Produces a reusable episode log (used by
# both Baseline A and Baseline B).

import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

_WEBSHOP_DIR = os.path.join(
    os.path.dirname(__file__), "..", "agent_system", "environments", "env_package", "webshop", "webshop"
)
if _WEBSHOP_DIR not in sys.path:
    sys.path.append(_WEBSHOP_DIR)

from orchestration.compare_specialist import CompareSpecialist
from orchestration.orchestrator import Orchestrator, OrchestratorAction
from orchestration.sub_agent import SubAgentResult

DEFAULT_ROUND_CAP = 6  # hard safety cap independent of any orchestrator policy


@dataclass
class EpisodeRound:
    round_idx: int
    orchestrator_action: OrchestratorAction
    orchestrator_raw_output: str
    sub_agent_result: Optional[Dict[str, Any]]  # None when action == "stop"


@dataclass
class EpisodeLog:
    task_description: str
    rounds: List[EpisodeRound] = field(default_factory=list)
    success: bool = False
    final_reward: float = 0.0
    num_delegation_rounds: int = 0
    stopped_by: str = "orchestrator"  # "orchestrator" | "round_cap" | "env_done"

    def to_dict(self) -> Dict[str, Any]:
        return {
            **{k: v for k, v in asdict(self).items() if k != "rounds"},
            "rounds": [asdict(r) for r in self.rounds],
        }


def make_env(seed: int = 0):
    import gym
    from web_agent_site.envs import WebAgentTextEnv  # noqa: F401 (import registers WebAgentTextEnv-v0 with gym)
    from web_agent_site.utils import DEBUG_PROD_SIZE

    return gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=DEBUG_PROD_SIZE)


def run_episode(
    env,
    orchestrator: Orchestrator,
    search_specialist,
    compare_specialist: CompareSpecialist,
    round_cap: int = DEFAULT_ROUND_CAP,
    orchestrator_override: Optional[List[OrchestratorAction]] = None,
) -> EpisodeLog:
    """Runs one full episode. `orchestrator_override`, if given, replaces the
    orchestrator's own decisions with a fixed action sequence -- this is how
    Baseline A (the fixed heuristic) reuses this exact same loop."""
    obs = env.reset()
    task_description = env.get_instruction_text() if hasattr(env, "get_instruction_text") else str(obs)
    avail_actions = env.get_available_actions()

    log = EpisodeLog(task_description=task_description)
    history: List[str] = []
    total_reward = 0.0

    for round_idx in range(round_cap):
        if orchestrator_override is not None and round_idx < len(orchestrator_override):
            action = orchestrator_override[round_idx]
            turn_raw = f"(overridden: {action})"
        else:
            turn = orchestrator.decide(task_description, round_idx, history, str(obs))
            action, turn_raw = turn.action, turn.raw_output

        if action == "stop":
            log.rounds.append(EpisodeRound(round_idx, action, turn_raw, None))
            log.stopped_by = "orchestrator"
            break

        specialist = search_specialist if action == "delegate-search" else compare_specialist
        result: SubAgentResult = specialist.run(env, task_description, str(obs), avail_actions)

        log.rounds.append(EpisodeRound(round_idx, action, turn_raw, asdict(result)))
        log.num_delegation_rounds += 1
        total_reward += result.total_reward
        history.append(f"{action} -> {len(result.turns)} step(s), reward={result.total_reward:.2f}")

        if result.turns:
            obs = result.final_observation
            avail_actions = env.get_available_actions()

        if result.env_done:
            log.stopped_by = "env_done"
            break
    else:
        log.stopped_by = "round_cap"

    log.final_reward = total_reward
    log.success = total_reward > 0
    return log
