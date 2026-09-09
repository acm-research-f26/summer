"""Phase 5: the error budget must be anisotropic, and honest about what it covers."""

import numpy as np
import pytest

from figomeas import load_config
from figomeas.uncertainty import (confidence_interval, ellipsoid_offset,
                                  jitter_end_slices, perturbation_draws,
                                  resample_fibroid)
from figomeas.synthetic import make_phantom

ANISO = (0.5, 0.5, 5.0)


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_inplane_perturbation_is_resolvable():
    a = np.zeros((60, 60, 6), bool)
    a[25:35, 25:35, 2:4] = True
    grown = ellipsoid_offset(a, ANISO, 1.0, 0.0)
    xs0 = np.ptp(np.nonzero(a)[0])
    xs1 = np.ptp(np.nonzero(grown)[0])
    assert xs1 - xs0 == 4              # +-1 mm at 0.5 mm/voxel


def test_erosion_shrinks_and_dilation_grows():
    a = np.zeros((60, 60, 6), bool)
    a[20:40, 20:40, 2:4] = True
    assert ellipsoid_offset(a, ANISO, 1.0, 0.0).sum() > a.sum()
    assert ellipsoid_offset(a, ANISO, -1.0, 0.0).sum() < a.sum()


def test_morphology_cannot_express_subslice_through_plane_shift():
    """Documents *why* the through-plane term is modelled at slice granularity.

    On a 5 mm grid every dilation below 5 mm is a no-op through-plane, so the
    +-2.5 mm boundary shift that dominates the error budget is invisible to
    morphology. If this ever stops being true the slice-jitter model should be
    revisited.
    """
    a = np.zeros((40, 40, 8), bool)
    a[15:25, 15:25, 3:5] = True
    for sub_slice in (1.0, 2.5, 4.0):
        assert ellipsoid_offset(a, ANISO, 0.0, sub_slice).sum() == a.sum()
    assert ellipsoid_offset(a, ANISO, 0.0, 5.0).sum() > a.sum()


def test_slice_jitter_moves_the_through_plane_extent():
    rng = np.random.default_rng(0)
    a = np.zeros((40, 40, 8), bool)
    a[15:25, 15:25, 3:5] = True
    extents = {int(jitter_end_slices(a, rng).any(axis=(0, 1)).sum()) for _ in range(200)}
    assert len(extents) > 1, "slice jitter never changed the extent"
    assert max(extents) > 2 and min(extents) < 2


def test_slice_jitter_leaves_the_interior_alone():
    rng = np.random.default_rng(1)
    a = np.zeros((30, 30, 9), bool)
    a[10:20, 10:20, 2:7] = True
    for _ in range(20):
        out = jitter_end_slices(a, rng)
        np.testing.assert_array_equal(out[:, :, 3:6], a[:, :, 3:6])


def test_draws_are_stratified_over_radius(cfg):
    draws = perturbation_draws(ANISO, cfg, np.random.default_rng(0))
    assert len(draws) == cfg["uncertainty.n_radius_draws"] * cfg["uncertainty.n_mask_draws"]
    assert len({d[1] for d in draws}) == cfg["uncertainty.n_radius_draws"]


def test_confidence_interval_shape(cfg):
    ci = confidence_interval(np.array([10., 20., 30., 40., 50.]), cfg)
    assert ci["ci_low"] <= ci["p_median"] <= ci["ci_high"]
    assert ci["ci_width"] == pytest.approx(ci["ci_high"] - ci["ci_low"])
    assert ci["n_samples"] == 5


@pytest.mark.parametrize("samples,expected", [
    ([45., 48., 52., 55.], True),
    ([10., 12., 14., 16.], False),
    ([80., 85., 90., 95.], False),
])
def test_straddle_detection(cfg, samples, expected):
    assert confidence_interval(np.array(samples), cfg)["ci_straddles_50"] is expected


def test_empty_samples_do_not_crash(cfg):
    ci = confidence_interval(np.array([]), cfg)
    assert ci["n_samples"] == 0 and np.isnan(ci["p_median"])


@pytest.mark.parametrize("figo_type", ["1", "2", "5", "6"])
def test_resampling_produces_a_band_around_the_point_estimate(cfg, figo_type):
    ph = make_phantom(figo_type)
    s = resample_fibroid(ph.data, ph.data == 3, ph.spacing, cfg, np.random.default_rng(0))
    ci = confidence_interval(s, cfg)
    assert ci["n_samples"] > 0
    assert 0 <= ci["ci_low"] <= ci["ci_high"] <= 100
    assert ci["ci_width"] > 0, "perturbation produced no spread at all"
