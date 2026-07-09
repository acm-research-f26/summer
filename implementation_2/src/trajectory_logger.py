"""
Trajectory logger for step-level browser agent evaluation.

Captures per-step actions, relevance scores, and timing so we can compare
baseline vs ARP-enhanced runs on the same benchmark tasks.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class StepRecord:
    step: int
    action: str
    target_text: str
    relevance_score: float | None
    in_top_k: bool | None
    duration_sec: float


@dataclass
class TrajectoryRecord:
    task_id: str
    mode: str  # "baseline" or "arp"
    prompt: str
    success: bool
    steps: int
    wall_time_sec: float
    irrelevant_clicks: int
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    step_records: list[StepRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrajectoryLogger:
    """Append-only logger for agent trajectories."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._current: TrajectoryRecord | None = None
        self._step_start: float | None = None

    def start_run(self, task_id: str, mode: str, prompt: str) -> None:
        self._current = TrajectoryRecord(
            task_id=task_id,
            mode=mode,
            prompt=prompt,
            success=False,
            steps=0,
            wall_time_sec=0.0,
            irrelevant_clicks=0,
        )
        self._step_start = time.monotonic()
        self._run_start = self._step_start

    def log_step(
        self,
        action: str,
        target_text: str,
        relevance_score: float | None = None,
        in_top_k: bool | None = None,
    ) -> None:
        if self._current is None:
            raise RuntimeError("No active run — call start_run() first")

        now = time.monotonic()
        duration = (now - self._step_start) if self._step_start else 0.0
        self._step_start = now

        self._current.steps += 1
        self._current.step_records.append(
            StepRecord(
                step=self._current.steps,
                action=action,
                target_text=target_text,
                relevance_score=relevance_score,
                in_top_k=in_top_k,
                duration_sec=round(duration, 3),
            )
        )

        if action == "click" and in_top_k is False:
            self._current.irrelevant_clicks += 1

    def finish_run(self, success: bool) -> TrajectoryRecord:
        if self._current is None:
            raise RuntimeError("No active run — call start_run() first")

        self._current.success = success
        self._current.wall_time_sec = round(time.monotonic() - self._run_start, 3)

        out_path = self.output_dir / f"{self._current.task_id}_{self._current.mode}.json"
        with open(out_path, "w") as f:
            json.dump(self._current.to_dict(), f, indent=2)

        record = self._current
        self._current = None
        return record
