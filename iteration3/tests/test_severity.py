import numpy as np
import pytest

from config import CATEGORIES, POTENCY_WEIGHTS, POTENCY_WEIGHTS_ALT
from src.severity import severity


def scores(**kwargs):
    s = {c: 0.0 for c in CATEGORIES}
    s.update(kwargs)
    return s


def test_all_zero_is_zero():
    assert severity(scores()) == 0.0


def test_all_ones_matches_noisy_or():
    expected = 1.0
    for c in CATEGORIES:
        expected_factor = 1 - POTENCY_WEIGHTS[c]
        expected *= expected_factor
    assert severity({c: 1.0 for c in CATEGORIES}) == pytest.approx(1 - expected)


def test_known_literal_value():
    # independent of config: explicit weights, hand-computed expectation
    weights = {"urgency": 0.2, "scarcity": 0.3, "social_proof": 0.5,
               "guilt_wording": 0.6, "other": 0.5}
    got = severity(scores(other=1.0, urgency=0.5), weights)
    assert got == pytest.approx(1 - (1 - 0.5) * (1 - 0.2 * 0.5))  # 0.55


def test_bounds_on_random_inputs():
    rng = np.random.default_rng(42)
    for _ in range(200):
        s = {c: float(rng.random()) for c in CATEGORIES}
        for weights in (POTENCY_WEIGHTS, POTENCY_WEIGHTS_ALT):
            v = severity(s, weights)
            assert 0.0 <= v <= 1.0


def test_monotonic_adding_a_pattern_never_lowers_severity():
    rng = np.random.default_rng(42)
    for _ in range(200):
        s = {c: float(rng.random()) for c in CATEGORIES}
        base = severity(s)
        bump_cat = CATEGORIES[int(rng.integers(len(CATEGORIES)))]
        bumped = dict(s)
        bumped[bump_cat] = min(1.0, s[bump_cat] + float(rng.random()) * (1 - s[bump_cat]))
        assert severity(bumped) >= base


def test_single_potent_pattern_scores_high():
    # noisy-OR rationale: one strong pattern is not diluted by absent ones
    assert severity(scores(other=1.0)) == pytest.approx(POTENCY_WEIGHTS["other"])
    assert severity(scores(other=1.0), POTENCY_WEIGHTS_ALT) == pytest.approx(0.9)


def test_works_on_both_systems_record_shapes():
    system_a_like = {"is_dark": 0.8, "categories": scores(urgency=0.7),
                     "top_category": "urgency"}
    system_b_like = {"is_dark": 0.5, "categories": scores(other=0.5),
                     "top_category": "other", "parse_failed": False,
                     "raw_response": "..."}
    for rec in (system_a_like, system_b_like):
        v = severity(rec["categories"])
        assert 0.0 < v <= 1.0


def test_missing_category_raises():
    bad = {c: 0.1 for c in CATEGORIES if c != "other"}
    with pytest.raises(KeyError):
        severity(bad)


def test_out_of_range_score_raises():
    with pytest.raises(ValueError):
        severity(scores(urgency=1.2))
    with pytest.raises(ValueError):
        severity(scores(urgency=-0.1))
