import numpy as np
import pandas as pd
import pytest

from config import CATEGORIES, POTENCY_WEIGHTS
from src.score_corpus import CORPUS_COLUMNS, build_corpus_table
from src.severity import severity


def cats(**kwargs):
    s = {c: 0.0 for c in CATEGORIES}
    s.update(kwargs)
    return s


@pytest.fixture
def fixture_inputs():
    truth = {
        "contextdp": pd.DataFrame([
            {"filename": "a.png", "path": "/img/a.png", "platform": "web",
             "urgency": 1, "scarcity": 0, "social_proof": 0,
             "guilt_wording": 0, "other": 0, "is_dark": 1},
            {"filename": "b.png", "path": "/img/b.png", "platform": "mobile",
             "urgency": 0, "scarcity": 0, "social_proof": 0,
             "guilt_wording": 0, "other": 1, "is_dark": 1},
        ]),
        "own": pd.DataFrame([
            {"filename": "FIVE_1.png", "path": "/img/FIVE_1.png",
             "platform": "own", "urgency": 0, "scarcity": 0,
             "social_proof": 0, "guilt_wording": 0, "other": 1, "is_dark": 1},
        ]),
    }
    preds_text = {
        "contextdp": {
            "a.png": {"is_dark": 0.9, "categories": cats(urgency=0.8),
                      "top_category": "urgency"},
            "b.png": {"is_dark": 0.2, "categories": cats(),
                      "top_category": "other"},
        },
        "own": {
            "FIVE_1.png": {"is_dark": 0.6, "categories": cats(other=0.4),
                           "top_category": "other"},
        },
    }
    preds_vision = {
        "contextdp": {
            "a.png": {"is_dark": 0.7, "categories": cats(urgency=0.6),
                      "top_category": "urgency", "parse_failed": False},
            # b.png deliberately missing -> the 2-uncached-screens case
        },
        "own": {
            "FIVE_1.png": {"is_dark": 0.5, "categories": cats(other=0.5),
                           "top_category": "other", "parse_failed": False},
        },
    }
    return truth, preds_text, preds_vision


def test_schema_and_counts(fixture_inputs):
    df = build_corpus_table(*fixture_inputs)
    assert list(df.columns) == CORPUS_COLUMNS
    assert len(df) == 3
    assert set(df["source_dataset"]) == {"contextdp", "own"}


def test_severities_match_direct_computation(fixture_inputs):
    truth, preds_text, preds_vision = fixture_inputs
    df = build_corpus_table(truth, preds_text, preds_vision).set_index("screen_id")
    a_text = preds_text["contextdp"]["a.png"]["categories"]
    assert df.loc["a.png", "severity_text"] == pytest.approx(
        severity(a_text, POTENCY_WEIGHTS)
    )
    assert df.loc["FIVE_1.png", "severity_vision"] == pytest.approx(
        severity(preds_vision["own"]["FIVE_1.png"]["categories"])
    )


def test_missing_vision_screen_becomes_nan_not_error(fixture_inputs):
    df = build_corpus_table(*fixture_inputs).set_index("screen_id")
    assert np.isnan(df.loc["b.png", "severity_vision"])
    assert np.isnan(df.loc["b.png", "vision_urgency"])
    # text side untouched
    assert not np.isnan(df.loc["b.png", "severity_text"])


def test_missing_text_screen_raises(fixture_inputs):
    truth, preds_text, preds_vision = fixture_inputs
    del preds_text["contextdp"]["b.png"]
    with pytest.raises(KeyError):
        build_corpus_table(truth, preds_text, preds_vision)
