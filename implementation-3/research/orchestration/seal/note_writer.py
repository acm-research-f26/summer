# IMP3 SEAL variant -- the self-edit generation policy (SEAL arXiv:2506.10943's
# "self-edit" step, specialized to this project: after a batch of episodes,
# the agent writes itself a short natural-language note about what it's
# learned about its own strategy). Structurally mirrors orchestration/
# orchestrator.py (uses its own adapter, has a system prompt, exposes a
# set_recorder() hook for training), but its output is free text wrapped in
# <note></note> tags rather than a categorical action.

from dataclasses import dataclass
from typing import Callable, List, Optional

from orchestration.model_pool import SELF_EDITOR_ADAPTER, GenerationConfig, generate, get_shared_model, use_adapter

NOTE_WRITER_SYSTEM_PROMPT = (
    "You are reflecting on a batch of recent ALFWorld task attempts by a sub-agent you are "
    "coaching. Based on the outcomes below, write ONE short, concrete strategy note (1-3 "
    "sentences) that would help the sub-agent do better on similar tasks next time -- e.g. "
    "a specific mistake pattern to avoid, or a heuristic that worked. Do not restate the task; "
    "give actionable strategy advice.\n"
    "Respond with EXACTLY one note, inside <note></note> tags, and nothing else."
)


def _summarize_episodes(episode_logs) -> str:
    lines = []
    for i, log in enumerate(episode_logs):
        outcome = "SUCCESS" if log.success else "FAILURE"
        lines.append(
            f"Episode {i}: task='{log.task_description}', outcome={outcome}, "
            f"delegation_rounds={log.num_delegation_rounds}, stopped_by={log.stopped_by}"
        )
    return "\n".join(lines) if lines else "(no episodes in this batch)"


@dataclass
class NoteCandidate:
    system_prompt: str
    user_prompt: str
    raw_output: str
    note: str  # empty string on malformed output -- treated as a no-op edit downstream


def _parse_note(raw: str) -> str:
    start, end = raw.find("<note>"), raw.find("</note>")
    if start == -1 or end == -1:
        return ""  # fail-safe: malformed output -> no-op edit, not a crash
    return raw[start + len("<note>"):end].strip()


class NoteWriter:
    def __init__(self, device: str = "cuda:0"):
        self.model, self.tokenizer = get_shared_model(device=device)
        self._recorder: Optional[Callable[[str, str, str], None]] = None

    def set_recorder(self, recorder: Optional[Callable[[str, str, str], None]]) -> None:
        self._recorder = recorder

    def write_note(self, recent_episode_logs: List, temperature: float = 0.9) -> NoteCandidate:
        user_prompt = (
            "Recent task attempts:\n" + _summarize_episodes(recent_episode_logs) +
            "\n\nWrite your strategy note now."
        )
        use_adapter(self.model, SELF_EDITOR_ADAPTER)
        cfg = GenerationConfig(max_new_tokens=96, temperature=temperature, do_sample=True)
        raw = generate(self.model, self.tokenizer, NOTE_WRITER_SYSTEM_PROMPT, user_prompt, cfg=cfg, response_prefix="<note>")
        if self._recorder is not None:
            self._recorder(NOTE_WRITER_SYSTEM_PROMPT, user_prompt, raw)
        return NoteCandidate(system_prompt=NOTE_WRITER_SYSTEM_PROMPT, user_prompt=user_prompt, raw_output=raw, note=_parse_note(raw))
