"""Phase 6: flagging logic, and the guards that keep the model honest."""

import numpy as np
import pandas as pd
import pytest

from figomeas import load_config
from figomeas.borderline import (PREDICTORS, assert_no_predictor_leakage,
                                 build_design_matrix, evaluate_grouped, flag)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def _table(n=240, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 100, n)
    return pd.DataFrame({
        "patient_id": [f"P{i//3:03d}" for i in range(n)],
        "component": np.arange(n) % 3 + 1,
        "percent_intramural": p,
        "ci_straddles_50": rng.random(n) < 0.15,
        "reliable": True,
        "touch_cavity": rng.random(n) < 0.4,
        "touch_serosa": rng.random(n) < 0.4,
        "volume_mm3": rng.lognormal(8, 1, n),
        "max_diameter_mm": rng.uniform(5, 90, n),
        "n_slices_spanned": rng.integers(1, 9, n),
        "elongation": rng.uniform(0.3, 1, n),
        "flatness": rng.uniform(0.3, 1, n),
        "solidity": rng.uniform(0.6, 1, n),
        "sphericity": rng.uniform(0.4, 1, n),
        "sx": 0.45, "sy": 0.45, "sz": 5.5,
    })


def test_point_estimate_inside_the_band_is_flagged(cfg):
    band = cfg["borderline.band_pct"]
    df = flag(pd.DataFrame({"percent_intramural": [50.0, 50 + band - 1, 50 + band + 5],
                            "ci_straddles_50": [False] * 3}), cfg)
    assert list(df.is_borderline) == [True, True, False]


def test_interval_crossing_fifty_is_flagged_even_when_far_from_it(cfg):
    df = flag(pd.DataFrame({"percent_intramural": [90.0], "ci_straddles_50": [True]}), cfg)
    assert bool(df.is_borderline.iloc[0])
    assert df.borderline_reason.iloc[0] == "interval"


def test_reason_records_both_causes(cfg):
    df = flag(pd.DataFrame({"percent_intramural": [50.0], "ci_straddles_50": [True]}), cfg)
    assert df.borderline_reason.iloc[0] == "point estimate and interval"


def test_missing_interval_column_degrades_gracefully(cfg):
    df = flag(pd.DataFrame({"percent_intramural": [50.0, 95.0]}), cfg)
    assert list(df.is_borderline) == [True, False]


# --------------------------------------------------------------- honesty guards

@pytest.mark.parametrize("leak", ["percent_intramural", "frac_intracavitary",
                                  "dist_to_cavity_mm", "ci_width", "figo",
                                  "touch_cavity", "p_median"])
def test_leaky_predictors_are_refused(leak):
    with pytest.raises(ValueError, match="leak"):
        assert_no_predictor_leakage(["volume_mm3", leak])


def test_declared_predictors_are_clean():
    assert_no_predictor_leakage(PREDICTORS)


def test_design_matrix_excludes_the_measurement(cfg):
    _, _, _, feats, _ = build_design_matrix(flag(_table(), cfg))
    assert "percent_intramural" not in feats
    assert all(not f.startswith("frac_") for f in feats)


def test_grouping_is_enforced_across_folds(cfg):
    """evaluate_grouped asserts internally; this proves the assert can fire."""
    metrics = evaluate_grouped(flag(_table(), cfg), cfg)
    assert len(metrics) == len(cfg["borderline.models"])
    assert metrics.roc_auc.between(0, 1).all()
    assert metrics.n.iloc[0] > 0


def test_every_fibroid_of_a_patient_shares_a_group(cfg):
    _, _, groups, _, d = build_design_matrix(flag(_table(), cfg))
    assert len(set(groups)) < len(groups), "grouping vector is not grouping anything"
    assert (pd.Series(groups).values == d.patient_id.values).all()
