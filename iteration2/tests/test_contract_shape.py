"""Both systems' screen-level outputs honor the shape: exactly the frozen
key names, floats in [0,1]. Uses fakes — no GPU, no model downloads.

Reminder: at screen level the category values are independent multi-label
presence probabilities (NOT a softmax) — same keys as iteration 1, different
semantics.
"""
import numpy as np
import pandas as pd
import pytest

from config import CATEGORIES
from src.aggregate import aggregate_snippets
from src.vlm_model import parse_vlm_response

SCREEN_KEYS = {"is_dark", "categories", "top_category"}


def assert_screen_shape(r: dict):
    assert SCREEN_KEYS <= set(r)
    assert isinstance(r["is_dark"], float) and 0.0 <= r["is_dark"] <= 1.0
    assert set(r["categories"]) == set(CATEGORIES)
    for v in r["categories"].values():
        assert isinstance(v, float) and 0.0 <= v <= 1.0
    assert r["top_category"] in CATEGORIES


def test_system_a_output_shape():
    it1_style = [{
        "is_dark": 0.93,
        "categories": {"urgency": 0.7, "scarcity": 0.1, "social_proof": 0.1,
                       "guilt_wording": 0.05, "other": 0.05},
        "top_category": "urgency",
    }]
    assert_screen_shape(aggregate_snippets(it1_style))
    assert_screen_shape(aggregate_snippets([]))  # no-text screen


def test_system_b_output_shape():
    r = parse_vlm_response(
        '{"is_dark": 0.4, "categories": {"urgency": 0.9, "scarcity": 0.8, '
        '"social_proof": 0, "guilt_wording": 0, "other": 0.3}}'
    )
    assert_screen_shape(r)
    # multi-label: two categories can both be high; no softmax constraint
    assert r["categories"]["urgency"] + r["categories"]["scarcity"] > 1.0


def test_score_system_end_to_end_with_fakes():
    from src.evaluate import score_system

    truth = pd.DataFrame([
        {"filename": "a.png", "platform": "web", "urgency": 1, "scarcity": 0,
         "social_proof": 0, "guilt_wording": 0, "other": 0, "is_dark": 1},
        {"filename": "b.png", "platform": "web", "urgency": 0, "scarcity": 0,
         "social_proof": 0, "guilt_wording": 0, "other": 0, "is_dark": 0},
    ])
    perfect = {
        "a.png": {"is_dark": 0.9, "top_category": "urgency",
                  "categories": {"urgency": 0.9, "scarcity": 0.0, "social_proof": 0.0,
                                 "guilt_wording": 0.0, "other": 0.0}},
        "b.png": {"is_dark": 0.1, "top_category": "other",
                  "categories": {c: 0.0 for c in CATEGORIES}},
    }
    m = score_system(truth, perfect)
    assert m["per_category"]["urgency"]["f1"] == 1.0
    assert m["is_dark_accuracy"] == 1.0
    # zero-support categories are n/a and excluded from macro
    assert m["per_category"]["guilt_wording"]["f1"] is None
    assert "guilt_wording" not in m["macro_over"]
    assert m["macro_f1"] == 1.0


def test_score_system_raises_on_missing_screen():
    from src.evaluate import score_system

    truth = pd.DataFrame([
        {"filename": "a.png", "platform": "web", "urgency": 1, "scarcity": 0,
         "social_proof": 0, "guilt_wording": 0, "other": 0, "is_dark": 1},
    ])
    with pytest.raises(KeyError):
        score_system(truth, {})
