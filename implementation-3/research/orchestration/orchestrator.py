# IMP3 -- the orchestrator: decides each round whether to delegate to the
# navigate specialist, the interact specialist, or stop the episode. This
# action space and prompt are new for this project -- they don't exist in
# the original HiPER-agent repo. Shares weights/adapter with the specialists
# by default (see model_pool.py); Baseline B (orchestration/baseline_b_rl_stop.py)
# is what actually trains this decision via GRPO.
#
# Ported from implementation-2's WebShop version: only this file's prose
# (system prompt, action descriptions) is environment-specific -- decide()/
# _parse_action() mechanics and the 3-way action-enum structure carried over
# unchanged, no env calls live in this file.

from dataclasses import dataclass
from typing import Callable, List, Literal, Optional

from orchestration.model_pool import ORCHESTRATOR_ADAPTER, GenerationConfig, generate, get_shared_model, use_adapter

OrchestratorAction = Literal["delegate-navigate", "delegate-interact", "stop"]

ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are the orchestrator for a multi-agent ALFWorld embodied-task assistant. "
    "You do not act in the environment directly -- you delegate each round to one "
    "of two specialists, or stop the episode:\n"
    "- 'delegate-navigate': hands control to a specialist that locates the object/receptacle the task needs "
    "(going to it, opening or examining it as needed).\n"
    "- 'delegate-interact': hands control to a specialist that manipulates the located object to satisfy the "
    "task (take/put/heat/cool/clean/toggle).\n"
    "- 'stop': ends the episode now (use this once you believe the task is done, or further delegation "
    "is unlikely to help).\n"
    "Respond with EXACTLY one action, inside <action></action> tags, and nothing else. "
    "The action MUST be one of: delegate-navigate, delegate-interact, stop.\n"
    "Example of a correctly formatted response: <action>delegate-navigate</action>"
)

VALID_ACTIONS = {"delegate-navigate", "delegate-interact", "stop"}

# GenerationConfig's own default (0.7) left the untrained model deterministically
# delegating until the round cap on every single rollout of a Baseline B group
# (avg_delegation_rounds=6.0, zero reward variance, zero GRPO gradient every
# round of a real full run -- see runs/EXPERIMENT_LOG.md run_id 20260719_131641)
# -- "stop" was apparently never sampled at all. Raised specifically for the
# orchestrator's decision (not globally -- specialists/note-writer keep the
# 0.7 default) to push some probability mass toward exploring "stop" so GRPO's
# groups can actually see reward variance to learn from.
ORCHESTRATOR_TEMPERATURE = 2.0


@dataclass
class OrchestratorTurn:
    round_idx: int
    task_description: str
    round_summary: str
    action: OrchestratorAction
    raw_output: str


def _build_user_prompt(task_description: str, round_idx: int, history: List[str], last_observation: str) -> str:
    history_text = "\n".join(f"Round {i}: {h}" for i, h in enumerate(history)) or "(none yet)"
    return (
        f"Task: {task_description}\n"
        f"This is round {round_idx}.\n"
        f"History of prior rounds:\n{history_text}\n"
        f"Current observation after the last round: {last_observation}\n"
        f"Decide your next action."
    )


def _parse_action(raw: str) -> OrchestratorAction:
    start, end = raw.find("<action>"), raw.find("</action>")
    if start == -1 or end == -1:
        return "stop"  # fail safe: malformed output ends the episode rather than looping forever
    candidate = raw[start + len("<action>"):end].strip().lower()
    return candidate if candidate in VALID_ACTIONS else "stop"


class Orchestrator:
    def __init__(self, device: str = "cuda:0"):
        self.model, self.tokenizer = get_shared_model(device=device)
        # Baseline B (GRPO training) needs the exact (system_prompt, user_prompt,
        # raw_output) of every decision to recompute log-probs later -- rather than
        # duplicating this class's episode-loop usage, it hooks in here via
        # set_recorder() and reuses episode_runner.run_episode unchanged.
        self._recorder: Optional[Callable[[str, str, str], None]] = None

    def set_recorder(self, recorder: Optional[Callable[[str, str, str], None]]) -> None:
        self._recorder = recorder

    def decide(self, task_description: str, round_idx: int, history: List[str], last_observation: str) -> OrchestratorTurn:
        user_prompt = _build_user_prompt(task_description, round_idx, history, last_observation)
        use_adapter(self.model, ORCHESTRATOR_ADAPTER)
        raw = generate(
            self.model, self.tokenizer, ORCHESTRATOR_SYSTEM_PROMPT, user_prompt,
            cfg=GenerationConfig(max_new_tokens=32, temperature=ORCHESTRATOR_TEMPERATURE),
            response_prefix="<action>",
        )
        if self._recorder is not None:
            self._recorder(ORCHESTRATOR_SYSTEM_PROMPT, user_prompt, raw)
        action = _parse_action(raw)
        return OrchestratorTurn(
            round_idx=round_idx, task_description=task_description,
            round_summary=user_prompt, action=action, raw_output=raw,
        )
