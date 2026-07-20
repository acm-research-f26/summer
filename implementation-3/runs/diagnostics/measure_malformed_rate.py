#!/usr/bin/env python
# IMP3 -- PROJECT_GUIDE.md section 12d / section 15 priority-4: nobody had
# measured how often the untrained 0.5B model's raw output actually fails to
# parse into a valid action, for either the orchestrator's <action> tag or the
# specialists' <switch>/<subgoal>/<action> triple. The full run under
# runs/run_all.sh (run_id 20260719_121243) came back at 0% success rate
# everywhere, with Baseline B showing EXACTLY zero gradient (pg_loss_mean=0.0,
# avg_delegation_rounds=0.0) on every one of 10 rounds -- consistent with the
# orchestrator's malformed-output fail-safe (defaults to "stop") firing
# deterministically regardless of the model's actual sampled text, which would
# erase the reward variance GRPO needs to learn anything. This script checks
# that theory directly against real ALFWorld tasks rather than guessing.
#
# Standalone, read-only: no training, no adapter updates, no results/ output --
# this is a one-off diagnostic, not part of the run_all.sh pipeline.

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_RUNS_DIR = os.path.dirname(_THIS_DIR)
sys.path.insert(0, _RUNS_DIR)
from common import RESEARCH_DIR  # noqa: F401,E402 -- import side effect: puts research/ on sys.path

from orchestration.alfworld_env import AlfworldSingleEpisodeEnv  # noqa: E402
from orchestration.episode_runner import _ALFWORLD_CONFIG  # noqa: E402
from orchestration.interact_specialist import InteractSpecialist  # noqa: E402
from orchestration.navigate_specialist import NavigateSpecialist  # noqa: E402
from orchestration.orchestrator import Orchestrator, _parse_action  # noqa: E402

N_ORCHESTRATOR_SAMPLES = 30
N_SPECIALIST_EPISODES = 10  # each contributes up to its step budget (5) of turns


def check_orchestrator(n: int) -> dict:
    orch = Orchestrator()
    raws = []
    malformed = 0
    action_counts = {"delegate-navigate": 0, "delegate-interact": 0, "stop": 0}
    for i in range(n):
        env = AlfworldSingleEpisodeEnv(_ALFWORLD_CONFIG, seed=i, is_train=True)
        obs = env.reset()
        task_description = env.get_instruction_text()
        turn = orch.decide(task_description, round_idx=0, history=[], last_observation=obs)
        raws.append(turn.raw_output)
        start, end = turn.raw_output.find("<action>"), turn.raw_output.find("</action>")
        if start == -1 or end == -1:
            malformed += 1
        # turn.action is already the parsed+fail-safe'd action (see _parse_action) --
        # tallying it (not just the malformed flag) is what actually answers "does
        # the orchestrator ever choose to stop", since a malformed sample also
        # counts as "stop" via the fail-safe.
        action_counts[turn.action] = action_counts.get(turn.action, 0) + 1
    return {
        "n": n,
        "malformed_rate": malformed / n,
        "action_distribution": action_counts,
        "sample_raw_outputs": raws[:5],
    }


def check_orchestrator_late_round(n: int) -> dict:
    """Same check, but at round_idx=5 with a fabricated history of 5 prior
    delegate-navigate rounds -- mimicking the actual state Baseline B's real
    rollouts were in right before hitting the round cap every time
    (avg_delegation_rounds=6.0 on every one of 10 rounds, see
    runs/EXPERIMENT_LOG.md run_id 20260719_131641). round_idx=0's fresh-task
    context is a weaker test of "does the model ever choose to stop" than this."""
    orch = Orchestrator()
    fake_history = [f"delegate-navigate -> 3 step(s), reward=0.00" for _ in range(5)]
    action_counts = {"delegate-navigate": 0, "delegate-interact": 0, "stop": 0}
    for i in range(n):
        env = AlfworldSingleEpisodeEnv(_ALFWORLD_CONFIG, seed=i, is_train=True)
        obs = env.reset()
        task_description = env.get_instruction_text()
        turn = orch.decide(task_description, round_idx=5, history=fake_history, last_observation=obs)
        action_counts[turn.action] = action_counts.get(turn.action, 0) + 1
    return {"n": n, "action_distribution": action_counts}


def check_specialists(n_episodes: int) -> dict:
    navigate = NavigateSpecialist()
    interact = InteractSpecialist()
    total_turns = 0
    invalid_turns = 0
    sample_invalid = []
    for i in range(n_episodes):
        env = AlfworldSingleEpisodeEnv(_ALFWORLD_CONFIG, seed=1000 + i, is_train=True)
        obs = env.reset()
        task_description = env.get_instruction_text()
        avail = env.get_available_actions()
        result = navigate.run(env, task_description, obs, avail)
        for t in result.turns:
            total_turns += 1
            if not t.valid:
                invalid_turns += 1
                if len(sample_invalid) < 5:
                    sample_invalid.append(t.raw_output)
        if not result.env_done:
            result2 = interact.run(env, task_description, result.final_observation, env.get_available_actions())
            for t in result2.turns:
                total_turns += 1
                if not t.valid:
                    invalid_turns += 1
                    if len(sample_invalid) < 5:
                        sample_invalid.append(t.raw_output)
    return {
        "n_episodes": n_episodes,
        "total_turns": total_turns,
        "invalid_rate": (invalid_turns / total_turns) if total_turns else None,
        "sample_invalid_raw_outputs": sample_invalid,
    }


def main():
    print("=== Orchestrator <action> malformed-output rate (round 0, fresh task) ===")
    orch_stats = check_orchestrator(N_ORCHESTRATOR_SAMPLES)
    print(f"n={orch_stats['n']}  malformed_rate={orch_stats['malformed_rate']:.2%}  "
          f"action_distribution={orch_stats['action_distribution']}")
    print("Sample raw outputs (first 5):")
    for r in orch_stats["sample_raw_outputs"]:
        print(f"  {r!r}")

    print()
    print("=== Orchestrator action distribution at round 5 (mimics Baseline B's real round-cap state) ===")
    late_stats = check_orchestrator_late_round(N_ORCHESTRATOR_SAMPLES)
    print(f"n={late_stats['n']}  action_distribution={late_stats['action_distribution']}")

    print()
    print("=== Specialist <switch>/<subgoal>/<action> invalid-action rate ===")
    spec_stats = check_specialists(N_SPECIALIST_EPISODES)
    print(f"episodes={spec_stats['n_episodes']}  total_turns={spec_stats['total_turns']}  "
          f"invalid_rate={spec_stats['invalid_rate']:.2%}" if spec_stats["invalid_rate"] is not None else "no turns collected")
    print("Sample invalid raw outputs (first 5):")
    for r in spec_stats["sample_invalid_raw_outputs"]:
        print(f"  {r!r}")


if __name__ == "__main__":
    main()
