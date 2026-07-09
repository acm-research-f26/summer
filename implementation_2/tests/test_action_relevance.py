#!/usr/bin/env python3
"""
Unit tests for the ARP scorer — run without browser or API keys.

  python -m pytest implementation_2/tests/test_action_relevance.py -v
  # or
  python implementation_2/tests/test_action_relevance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.sample_pages import UTD_HOMEPAGE_ELEMENTS
from src.action_relevance import ActionRelevanceScorer


def test_admissions_task_ranks_admissions_first():
    task = "Open the UTD website and find when admissions open for fall semester"
    scorer = ActionRelevanceScorer(top_k=3)
    ranked = scorer.rank(task, UTD_HOMEPAGE_ELEMENTS)

    assert len(ranked) >= 2
    top_texts = [r.text.lower() for r in ranked[:3]]
    assert any("admission" in t or "apply" in t for t in top_texts), f"Got: {top_texts}"


def test_home_link_scores_lower_than_admissions():
    task = "find admissions deadline for fall semester"
    scorer = ActionRelevanceScorer(top_k=5)
    ranked = scorer.rank(task, UTD_HOMEPAGE_ELEMENTS)

    scores = {r.index: r.score for r in ranked}
    if 0 in scores and 3 in scores:
        assert scores[3] > scores[0], "Admissions should outrank Home"


def test_empty_elements_returns_empty():
    scorer = ActionRelevanceScorer()
    assert scorer.rank("any task", []) == []


if __name__ == "__main__":
    test_admissions_task_ranks_admissions_first()
    test_home_link_scores_lower_than_admissions()
    test_empty_elements_returns_empty()
    print("All tests passed.")
