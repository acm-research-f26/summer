"""
Observation enhancer — injects ARP-ranked element hints into agent context.

The browser-use agent sees a flat list of interactive elements. This module
prepends a compact relevance summary so the LLM can prioritize high-scoring
targets and avoid wasted clicks on navigation chrome.
"""

from __future__ import annotations

from .action_relevance import ActionRelevanceScorer, ScoredElement


ARP_HEADER = "[Action Relevance Prior — top elements for this task]"


def format_arp_hint(task: str, ranked: list[ScoredElement]) -> str:
    """Format ranked elements as a short context block for the LLM."""
    if not ranked:
        return ""

    lines = [ARP_HEADER, f'Task: "{task}"', "Ranked interactive elements:"]
    for i, el in enumerate(ranked, 1):
        label = el.text[:80] + ("…" if len(el.text) > 80 else "")
        lines.append(f"  {i}. [{el.index}] <{el.tag}> {label!r}  (relevance={el.score:.3f})")
    lines.append("Prefer elements with higher relevance scores when choosing actions.\n")
    return "\n".join(lines)


class ObservationEnhancer:
    """Wraps ActionRelevanceScorer and produces augmented observations."""

    def __init__(self, scorer: ActionRelevanceScorer | None = None):
        self.scorer = scorer or ActionRelevanceScorer()

    def enhance(self, task: str, observation: str, elements: list[dict]) -> tuple[str, list[ScoredElement]]:
        ranked = self.scorer.rank(task, elements)
        hint = format_arp_hint(task, ranked)
        if not hint:
            return observation, ranked
        return f"{hint}\n{observation}", ranked
