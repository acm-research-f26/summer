"""Phase 7: slice decimation must degrade the measurement in the expected direction."""

import numpy as np
import pytest

from figomeas import load_config
from figomeas.figo import derive
from figomeas.robustness import (_match_by_rank, decimate_slices,
                                 measure_under_decimation,
                                 reconstruction_sensitivity, slice_count_strata)
from figomeas.synthetic import make_phantom
import pandas as pd


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_decimation_keeps_every_nth_slice():
    a = np.zeros((10, 10, 12), bool)
    a[..., ::1] = True
    thin, mult = decimate_slices(a, 3)
    assert thin.shape[2] == 4 and mult == 3


def test_decimation_factor_one_is_identity():
    a = np.random.default_rng(0).random((6, 6, 9)) > 0.5
    thin, mult = decimate_slices(a, 1)
    np.testing.assert_array_equal(thin, a)
    assert mult == 1


def test_decimation_preserves_physical_thickness(cfg):
    """Dropping every 2nd slice must double the spacing, not shrink the uterus."""
    ph = make_phantom("4")
    thin, mult = decimate_slices(ph.data, 2)
    assert thin.shape[2] == (ph.data.shape[2] + 1) // 2
    assert mult * ph.spacing[2] == 2 * ph.spacing[2]


def test_matching_pairs_by_volume_rank():
    base = [{"volume_mm3": 10.0}, {"volume_mm3": 100.0}]
    other = [{"volume_mm3": 90.0}, {"volume_mm3": 9.0}]
    pairs = _match_by_rank(base, other)
    assert pairs[0][0]["volume_mm3"] == 100.0 and pairs[0][1]["volume_mm3"] == 90.0


def test_matching_drops_unpaired_components():
    assert len(_match_by_rank([{"volume_mm3": 1.0}] * 3, [{"volume_mm3": 1.0}])) == 1


@pytest.mark.parametrize("figo_type", ["2", "5"])
def test_decimation_moves_the_measurement(cfg, figo_type):
    """If halving the slices changed nothing, the study would be measuring nothing."""
    ph = make_phantom(figo_type)
    full = measure_under_decimation(ph.data, ph.spacing, cfg, 1)
    half = measure_under_decimation(ph.data, ph.spacing, cfg, 2)
    assert full and half
    assert full[0]["n_slices_spanned"] > half[0]["n_slices_spanned"]


def test_reconstruction_sensitivity_spans_the_scales(cfg):
    ph = make_phantom("5")
    out = reconstruction_sensitivity(ph.data, ph.data == 3, ph.spacing, cfg, [1.0, 1.5, 2.0])
    assert len(out) == 3
    assert out.percent_intramural.between(0, 100).all()
    assert out.percent_intramural.nunique() > 1, "bridging radius had no effect at all"


def test_slice_count_strata_shrink_monotonically():
    df = pd.DataFrame({"percent_intramural": np.linspace(0, 100, 50),
                       "n_slices_spanned": np.tile([1, 3, 5, 7, 9], 10)})
    out = slice_count_strata(df)
    assert list(out.subset) == ["all fibroids", ">= 4 slices", ">= 6 slices"]
    assert out.n.iloc[0] >= out.n.iloc[1] >= out.n.iloc[2]


def test_isotropic_subset_claim_is_refuted_by_the_manifest():
    """PLAN.md F2: the 33 'isotropic' patients are an unset-pixdim artifact."""
    from figomeas.config import REPO_ROOT
    man_path = REPO_ROOT / "results" / "manifest.csv"
    if not man_path.exists():
        pytest.skip("manifest not built")
    man = pd.read_csv(man_path)
    assert (man.anisotropy_ratio >= 2.0).all(), "an isotropic subset would appear here"
    seq = man[man.filename_kind == "seq"]
    assert (seq.nifti_sz == 1.0).all()          # what the header claimed
    assert (seq.used_sz > 3.0).all()            # what DICOM says is true
