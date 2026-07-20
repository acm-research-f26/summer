# IMP3 -- wires the orchestrator + two specialists together against a
# live ALFWorld env for one full episode: orchestrator decides, delegates,
# sub-agent executes (Part 1's mechanism), result returns to orchestrator,
# repeat until stop or a round cap. Produces a reusable episode log (used by
# both Baseline A and Baseline B).
#
# Ported from implementation-2's WebShop version: only make_env() changes
# (ALFWorld's package is already importable once implementation's root is on
# sys.path -- no manual sys.path append like WebShop's vendored web_agent_site
# needed). run_episode()'s loop mechanics are unchanged; only the specialist
# parameter names/imports and the delegate-* action strings follow from the
# orchestrator's renamed action enum.

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import verl  # noqa: F401 -- imported only to locate implementation-1/ via its install path below

# agent_system lives in implementation-1/, a sibling of implementation-2/ (this folder), not a
# subdirectory of it -- so we can't derive its path from this file's own location. verl is
# installed editable from implementation-1/'s root, so its resolved location is a reliable anchor
# regardless of where each implementation folder actually sits on disk.
_IMPLEMENTATION_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(verl.__file__)))
_ALFWORLD_CONFIG = os.path.join(
    _IMPLEMENTATION_ROOT, "agent_system", "environments", "env_package",
    "alfworld", "configs", "config_tw.yaml",
)

from orchestration.interact_specialist import InteractSpecialist
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


def make_env(seed: int = 0, is_train: bool = True):
    from orchestration.alfworld_env import AlfworldSingleEpisodeEnv

    return AlfworldSingleEpisodeEnv(_ALFWORLD_CONFIG, seed=seed, is_train=is_train)


def run_episode(
    env,
    orchestrator: Orchestrator,
    navigate_specialist,
    interact_specialist: InteractSpecialist,
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

        specialist = navigate_specialist if action == "delegate-navigate" else interact_specialist
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
