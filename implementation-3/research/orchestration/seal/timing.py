# IMP3 SEAL variant -- wall-clock overhead tracking (SEAL's own paper warning
# #2: each self-edit round adds a fine-tuning step Baseline B's plain GRPO
# round doesn't have). A small timed-phase utility, reused by
# sub_agent_trainer.py's and baseline_b_rl_stop.py's round loops too (not
# just SEAL's) so compare_variants.py's final overhead ratio is measured
# apples-to-apples from the same instrument, not hand-estimated.

import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RoundTiming:
    round_idx: int
    phases: Dict[str, float] = field(default_factory=dict)

    @property
    def total_s(self) -> float:
        return sum(self.phases.values())


class RoundTimer:
    """Usage:
        rt = RoundTimer()
        for round_idx in range(num_rounds):
            with rt.round(round_idx):
                with rt.phase("episode_collection"):
                    ...
                with rt.phase("note_generation"):
                    ...
    """

    def __init__(self):
        self.rounds: List[RoundTiming] = []
        self._current: Optional[RoundTiming] = None

    @contextmanager
    def round(self, round_idx: int):
        self._current = RoundTiming(round_idx=round_idx)
        try:
            yield self._current
        finally:
            self.rounds.append(self._current)
            self._current = None

    @contextmanager
    def phase(self, name: str):
        if self._current is None:
            raise RuntimeError("phase() used outside of a round() context")
        start = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - start
            self._current.phases[name] = self._current.phases.get(name, 0.0) + elapsed

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            for r in self.rounds:
                f.write(json.dumps({"round_idx": r.round_idx, **r.phases, "total_s": r.total_s}) + "\n")

    def mean_total_s(self) -> float:
        return sum(r.total_s for r in self.rounds) / len(self.rounds) if self.rounds else 0.0
