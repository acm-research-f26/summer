"""Frozen-contract test for src.inference.predict / predict_batch."""
from __future__ import annotations

import math

from src.inference import predict, predict_batch

EXPECTED_CATEGORIES = {
    "urgency", "scarcity", "social_proof", "guilt_wording", "other",
}


def _check(result):
    assert set(result.keys()) == {"is_dark", "categories", "top_category"}, result

    assert isinstance(result["is_dark"], float), type(result["is_dark"])
    assert 0.0 <= result["is_dark"] <= 1.0, result["is_dark"]

    cats = result["categories"]
    assert set(cats.keys()) == EXPECTED_CATEGORIES, set(cats.keys())
    for k, v in cats.items():
        assert isinstance(v, float), (k, type(v))
        assert 0.0 <= v <= 1.0, (k, v)
    assert math.isclose(sum(cats.values()), 1.0, abs_tol=1e-5), sum(cats.values())

    assert isinstance(result["top_category"], str)
    assert result["top_category"] in EXPECTED_CATEGORIES
    assert result["top_category"] == max(cats, key=cats.get)


def test_predict_single():
    _check(predict("Only 2 left in stock!"))


def test_predict_batch():
    out = predict_batch([
        "Only 2 left in stock!",
        "No thanks, I prefer to pay full price.",
    ])
    assert isinstance(out, list)
    assert len(out) == 2
    for r in out:
        _check(r)
