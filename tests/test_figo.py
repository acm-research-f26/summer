"""Phase 4: every FIGO rule asserted against geometry known by construction.

Two layers. First the rule table is exercised directly with hand-built records,
so a threshold typo is caught without running any morphology. Then the whole
pipeline is run end to end on synthetic phantoms, so a rule that is individually
correct but fed the wrong measurement still fails.
"""

import pytest

from figomeas import load_config
from figomeas.features import extract_patient, partition_residual
from figomeas.figo import (SUBMUCOSAL, SUBSEROSAL, components, derive,
                           is_submucosal, is_subserosal, near_threshold)
from figomeas.synthetic import all_phantoms, make_phantom


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def rec(p, cav=0.0, ser=0.0, tc=False, ts=False):
    return {"percent_intramural": p, "frac_intracavitary": cav,
            "frac_extraserosal": ser, "touch_cavity": tc, "touch_serosa": ts}


# ------------------------------------------------------------------ rule table

@pytest.mark.parametrize("record,expected", [
    (rec(2, cav=98, tc=True), "0"),
    (rec(30, cav=70, tc=True), "1"),
    (rec(80, cav=20, tc=True), "2"),
    (rec(100, cav=0, tc=True), "3"),
    (rec(100), "4"),
    (rec(80, ser=20, ts=True), "5"),
    (rec(30, ser=70, ts=True), "6"),
    (rec(2, ser=98, ts=True), "7"),
])
def test_each_figo_type(cfg, record, expected):
    assert derive(record, cfg) == expected


@pytest.mark.parametrize("p,expected", [(49.9, "1"), (50.0, "2"), (50.1, "2")])
def test_the_fifty_percent_boundary_is_inclusive_upward(cfg, p, expected):
    """1-vs-2 is the clinically load-bearing call; >=50 is type 2 by definition."""
    assert derive(rec(p, cav=100 - p, tc=True), cfg) == expected


@pytest.mark.parametrize("p,expected", [(49.9, "6"), (50.0, "5"), (50.1, "5")])
def test_the_boundary_for_subserosal(cfg, p, expected):
    assert derive(rec(p, ser=100 - p, ts=True), cfg) == expected


def test_type_3_needs_no_cavity_bulge_not_merely_high_p(cfg):
    """Regression: a type 2 with a 4% bulge was derived as type 3."""
    assert derive(rec(95.7, cav=4.3, tc=True), cfg) == "2"
    assert derive(rec(100.0, cav=0.0, tc=True), cfg) == "3"


@pytest.mark.parametrize("p,expected", [(60, "2-5"), (40, "1-6")])
def test_hybrids_report_both_families(cfg, p, expected):
    got = derive(rec(p, cav=20, ser=20, tc=True, ts=True), cfg)
    assert got == expected
    assert len(components(got, cfg)) == 2


def test_hybrid_counts_as_both_families(cfg):
    assert is_submucosal("2-5", cfg) and is_subserosal("2-5", cfg)


def test_nan_measurement_is_not_assigned_a_type(cfg):
    assert derive(rec(float("nan")), cfg) == "NA"


def test_type_8_is_never_derived(cfg):
    """Type 8 needs a cervix landmark UMD does not label; guessing it would be fiction."""
    got = {derive(rec(p, cav=c, ser=s, tc=tc, ts=ts), cfg)
           for p in (0, 25, 50, 75, 100) for c in (0, 50) for s in (0, 50)
           for tc in (True, False) for ts in (True, False)}
    assert not any("8" in g for g in got)


def test_near_threshold_band(cfg):
    band = cfg["borderline.band_pct"]
    assert near_threshold(rec(50), cfg)
    assert near_threshold(rec(50 + band), cfg)
    assert not near_threshold(rec(50 + band + 1), cfg)


def test_family_sets_are_disjoint():
    assert not (SUBMUCOSAL & SUBSEROSAL)


# ------------------------------------------------------------------ end to end

@pytest.mark.parametrize("figo_type", sorted(all_phantoms().keys()))
def test_phantom_derives_its_own_figo_type(cfg, figo_type):
    """The full pipeline -- reconstruct, measure, derive -- on known geometry."""
    ph = make_phantom(figo_type)
    records, _, _ = extract_patient(ph.data, ph.spacing, cfg, "phantom")
    assert len(records) == 1
    assert derive(records[0], cfg) == figo_type


@pytest.mark.parametrize("figo_type", sorted(all_phantoms().keys()))
def test_the_three_fractions_partition_the_fibroid(cfg, figo_type):
    """intramural + intracavitary + extraserosal must be exactly 100%."""
    ph = make_phantom(figo_type)
    records, _, _ = extract_patient(ph.data, ph.spacing, cfg, "phantom")
    assert partition_residual(records[0]) < 1e-6


def test_size_filter_drops_specks_and_reports_them(cfg):
    ph = make_phantom("4")
    data = ph.data.copy()
    data[2, 2, 1] = 3                       # one-voxel speck, far below the floor
    records, dropped, n_raw = extract_patient(data, ph.spacing, cfg, "phantom")
    assert n_raw == 2 and dropped == 1 and len(records) == 1
