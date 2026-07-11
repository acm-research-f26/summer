# IMP2 Part 2 -- the orchestrator: decides each round whether to delegate to
# the search specialist, the compare specialist, or stop the episode. This
# action space and prompt are new for this project -- they don't exist in
# the original HiPER-agent repo. Shares weights/adapter with the specialists
# by default (see model_pool.py); Baseline B (orchestration/baseline_b_rl_stop.py)
# is what actually trains this decision via GRPO.

from dataclasses import dataclass
from typing import Callable, List, Literal, Optional

from orchestration.model_pool import ORCHESTRATOR_ADAPTER, GenerationConfig, generate, get_shared_model, use_adapter

OrchestratorAction = Literal["delegate-search", "delegate-compare", "stop"]

ORCHESTRATOR_SYSTEM_PROMPT = (
    "You are the orchestrator for a multi-agent WebShop shopping assistant. "
    "You do not act in the environment directly -- you delegate each round to one "
    "of two specialists, or stop the episode:\n"
    "- 'delegate-search': hands control to a specialist that searches for and opens candidate products.\n"
    "- 'delegate-compare': hands control to a specialist that compares candidate products and buys the best match.\n"
    "- 'stop': ends the episode now (use this once you believe the task is done, or further delegation "
    "is unlikely to help).\n"
    "Respond with EXACTLY one action, inside <action></action> tags, and nothing else. "
    "The action MUST be one of: delegate-search, delegate-compare, stop."
)

VALID_ACTIONS = {"delegate-search", "delegate-compare", "stop"}


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
        raw = generate(self.model, self.tokenizer, ORCHESTRATOR_SYSTEM_PROMPT, user_prompt, cfg=GenerationConfig(max_new_tokens=32))
        if self._recorder is not None:
            self._recorder(ORCHESTRATOR_SYSTEM_PROMPT, user_prompt, raw)
        action = _parse_action(raw)
        return OrchestratorTurn(
            round_idx=round_idx, task_description=task_description,
            round_summary=user_prompt, action=action, raw_output=raw,
        )
