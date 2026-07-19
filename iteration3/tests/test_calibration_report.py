import numpy as np
import pandas as pd
import pytest

from config import CATEGORIES, POTENCY_WEIGHTS
from src.calibration_report import (
    comparison_table,
    detection_table,
    hedge_table,
    severity_summary,
)
from src.severity import severity


def _corpus_row(screen_id, vision_scores, gt=0):
    """Minimal corpus_scores row (v4 vision side) for fixtures."""
    row = {
        "screen_id": screen_id,
        "source_dataset": "own",
        "gt_is_dark": gt,
        "vision_is_dark": max(vision_scores.values()),
    }
    for c in CATEGORIES:
        row[f"gt_{c}"] = gt
        row[f"vision_{c}"] = vision_scores[c]
    row["severity_vision"] = severity(vision_scores, POTENCY_WEIGHTS)
    return row


def _baseline_rec(scores):
    return {"is_dark": max(scores.values()), "categories": dict(scores)}


@pytest.fixture
def fixtures():
    hedged = {c: 0.5 for c in CATEGORIES}
    confident = {**{c: 0.0 for c in CATEGORIES}, "other": 0.9}
    zero = {c: 0.0 for c in CATEGORIES}

    corpus = pd.DataFrame([
        _corpus_row("a.png", confident, gt=1),
        _corpus_row("b.png", zero, gt=0),
        # scored by v4 but absent from the v3 baseline -> dropped from cmp
        _corpus_row("v4only.png", confident, gt=1),
    ])
    baseline = {
        "a.png": _baseline_rec(hedged),
        "b.png": _baseline_rec(hedged),
        "notincorpus.png": _baseline_rec(zero),
    }
    return corpus, baseline


def test_comparison_table_inner_joins(fixtures):
    corpus, baseline = fixtures
    cmp = comparison_table(corpus, baseline)
    assert sorted(cmp["screen_id"]) == ["a.png", "b.png"]
    # v3 side comes from the baseline record, v4 side from the corpus
    assert (cmp.set_index("screen_id").loc["a.png", "v3_other"]) == 0.5
    assert (cmp.set_index("screen_id").loc["a.png", "v4_other"]) == 0.9
    # severity_v3 recomputed from baseline categories via the same formula
    expected = severity({c: 0.5 for c in CATEGORIES}, POTENCY_WEIGHTS)
    assert cmp.set_index("screen_id").loc["a.png", "severity_v3"] == pytest.approx(expected)


def test_comparison_table_skips_nan_vision(fixtures):
    corpus, baseline = fixtures
    corpus.loc[corpus["screen_id"] == "a.png",
               [f"vision_{c}" for c in CATEGORIES]] = np.nan
    cmp = comparison_table(corpus, baseline)
    assert cmp["screen_id"].tolist() == ["b.png"]


def test_hedge_table_counts_the_parked_mass(fixtures):
    corpus, baseline = fixtures
    hedge = hedge_table(comparison_table(corpus, baseline)).set_index("category")
    # v3: every category score on both screens is exactly 0.5
    assert hedge.loc["ALL", "v3_at_0.5"] == 1.0
    assert hedge.loc["ALL", "v3_in_band"] == 1.0
    # v4: no score anywhere near the band
    assert hedge.loc["ALL", "v4_at_0.5"] == 0.0
    assert hedge.loc["ALL", "v4_in_band"] == 0.0


def test_detection_table_prf(fixtures):
    corpus, baseline = fixtures
    det = detection_table(comparison_table(corpus, baseline)).set_index("category")
    # gt: a.png positive for every category, b.png negative.
    # v3 predicts 0.5>=0.5 on both screens: recall 1, precision 0.5
    assert det.loc["other", "v3_precision"] == 0.5
    assert det.loc["other", "v3_recall"] == 1.0
    # v4 fires only on a.png/other: perfect for other...
    assert det.loc["other", "v4_f1"] == 1.0
    # ...and misses the other categories entirely (recall 0)
    assert det.loc["urgency", "v4_recall"] == 0.0


def test_severity_summary_range(fixtures):
    corpus, baseline = fixtures
    sev = severity_summary(comparison_table(corpus, baseline)).set_index("score")
    assert sev.loc["severity_v3", "range"] == 0.0  # both screens identical under v3
    assert sev.loc["severity_v4", "range"] > 0.3
