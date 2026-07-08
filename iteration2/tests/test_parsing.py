"""Defensive VLM response parsing: fences, prose, malformed, regex fallback."""
from config import CATEGORIES
from src.vlm_model import parse_vlm_response

CLEAN = ('{"is_dark": 0.9, "categories": {"urgency": 0.8, "scarcity": 0.1, '
         '"social_proof": 0.0, "guilt_wording": 0.0, "other": 0.2}}')


def _check_shape(r):
    assert set(r) == {"is_dark", "categories", "top_category"}
    assert set(r["categories"]) == set(CATEGORIES)
    assert 0.0 <= r["is_dark"] <= 1.0
    assert all(0.0 <= v <= 1.0 for v in r["categories"].values())


def test_clean_json():
    r = parse_vlm_response(CLEAN)
    _check_shape(r)
    assert r["is_dark"] == 0.9
    assert r["top_category"] == "urgency"


def test_markdown_fences_stripped():
    r = parse_vlm_response(f"```json\n{CLEAN}\n```")
    _check_shape(r)
    assert r["categories"]["urgency"] == 0.8


def test_surrounding_prose_tolerated():
    r = parse_vlm_response(f"Sure! Here is my analysis:\n{CLEAN}\nHope that helps.")
    _check_shape(r)


def test_missing_keys_default_to_zero():
    r = parse_vlm_response('{"is_dark": 0.7, "categories": {"urgency": 0.6}}')
    _check_shape(r)
    assert r["categories"]["scarcity"] == 0.0


def test_out_of_range_values_clamped():
    r = parse_vlm_response('{"is_dark": 1.7, "categories": {"urgency": -0.2}}')
    _check_shape(r)
    assert r["is_dark"] == 1.0
    assert r["categories"]["urgency"] == 0.0


def test_regex_fallback_on_broken_json():
    raw = 'urgency: 0.7, scarcity: 0.1 and "is_dark": 0.85 — not valid JSON {'
    r = parse_vlm_response(raw)
    _check_shape(r)
    assert r["categories"]["urgency"] == 0.7
    assert r["is_dark"] == 0.85


def test_hopeless_output_returns_none():
    assert parse_vlm_response("I cannot analyze this image.") is None
