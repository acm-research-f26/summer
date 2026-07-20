# IMP3 SEAL variant -- catastrophic-forgetting tracking (SEAL's own paper
# warning #1). Fixes a small held-out set of ALFWorld tasks BEFORE any
# self-edits are applied, then re-evaluates success rate on that exact set
# every N rounds so a forgetting curve can be reported from real numbers, not
# asserted. A flat/declining held-out curve alongside rising in-distribution
# performance is the forgetting signal to watch for.

import json
from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class ForgettingCheckpoint:
    round_idx: int
    held_out_success_rate: float
    num_tasks: int


class ForgettingTracker:
    def __init__(self, held_out_seeds: List[int], config_path: str, check_every_n_rounds: int = 2):
        # `held_out_seeds` must not overlap with the seeds used for training
        # rounds/eval batches elsewhere -- callers are responsible for picking
        # a disjoint range (seal_trainer.py uses a distinct high seed range).
        self.held_out_seeds = held_out_seeds
        self.config_path = config_path
        self.check_every_n_rounds = check_every_n_rounds
        self.checkpoints: List[ForgettingCheckpoint] = []

    def should_check(self, round_idx: int) -> bool:
        return round_idx % self.check_every_n_rounds == 0

    def evaluate(self, round_idx: int, orchestrator, navigate_specialist, interact_specialist) -> ForgettingCheckpoint:
        from orchestration.alfworld_env import AlfworldSingleEpisodeEnv
        from orchestration.episode_runner import run_episode

        # 3 full navigate+interact cycles, not 1 -- see baseline_a_heuristic.py's
        # FIXED_ACTION_SEQUENCE comment. Matches run_episode's default round_cap=6
        # exactly (not passed explicitly here) so the override never runs out.
        FIXED_DELEGATION_SEQUENCE = [
            "delegate-navigate", "delegate-interact",
            "delegate-navigate", "delegate-interact",
            "delegate-navigate", "delegate-interact",
        ]
        successes = []
        for seed in self.held_out_seeds:
            env = AlfworldSingleEpisodeEnv(self.config_path, seed=seed, is_train=True)
            log = run_episode(
                env, orchestrator, navigate_specialist, interact_specialist,
                orchestrator_override=FIXED_DELEGATION_SEQUENCE,
            )
            successes.append(log.success)
        rate = sum(successes) / len(successes) if successes else 0.0
        cp = ForgettingCheckpoint(round_idx=round_idx, held_out_success_rate=rate, num_tasks=len(successes))
        self.checkpoints.append(cp)
        return cp

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump([asdict(cp) for cp in self.checkpoints], f, indent=2)
