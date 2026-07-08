"""Aggregation rule (max_product) on hand-built snippet outputs. No GPU."""
from config import CATEGORIES
from src.aggregate import aggregate_snippets, empty_screen_result


def _snippet(is_dark, **cats):
    categories = {c: 0.0 for c in CATEGORIES}
    categories.update(cats)
    return {"is_dark": is_dark, "categories": categories,
            "top_category": max(categories, key=categories.get)}


def test_empty_screen_is_all_zeros():
    r = aggregate_snippets([])
    assert r == empty_screen_result()
    assert r["is_dark"] == 0.0
    assert all(v == 0.0 for v in r["categories"].values())


def test_one_dark_banner_flags_screen_despite_benign_blocks():
    benign = _snippet(0.05, other=0.9)
    banner = _snippet(0.95, urgency=0.9)
    r = aggregate_snippets([benign] * 40 + [banner])
    assert r["is_dark"] == 0.95
    assert abs(r["categories"]["urgency"] - 0.95 * 0.9) < 1e-9
    assert r["top_category"] == "urgency"


def test_benign_softmax_mass_is_suppressed():
    # a clearly-benign block whose softmax happens to lean 'scarcity'
    benign = _snippet(0.02, scarcity=0.8)
    r = aggregate_snippets([benign])
    assert r["categories"]["scarcity"] <= 0.02 * 0.8 + 1e-9
    assert r["is_dark"] == 0.02


def test_multi_label_screen_keeps_both_categories():
    r = aggregate_snippets([_snippet(0.9, urgency=0.8), _snippet(0.9, scarcity=0.7)])
    assert r["categories"]["urgency"] > 0.5
    assert r["categories"]["scarcity"] > 0.5  # independent, not softmax
