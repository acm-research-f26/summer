"""Phase 0: the config is the single source of truth for every threshold."""

import pytest

from figomeas import load_config


@pytest.fixture(scope="module")
def cfg():
    return load_config()


def test_config_loads(cfg):
    assert cfg.path.exists()
    assert "figo" in cfg


def test_dotted_access(cfg):
    assert cfg["figo.intramural_threshold_pct"] == 50.0
    assert cfg["labels.fibroid"] == 3


def test_missing_key_raises(cfg):
    with pytest.raises(KeyError):
        cfg["figo.no_such_key"]
    assert cfg.get("figo.no_such_key", "fallback") == "fallback"


@pytest.mark.parametrize(
    "key",
    [
        "seed",
        "data.root",
        "data.mask_glob",
        "labels.wall",
        "labels.cavity",
        "labels.fibroid",
        "spacing.source_priority",
        "body.method",
        "body.closing_radius_mm",
        "body.max_volume_ratio_vs_union",
        "fibroid.min_volume_mm3",
        "figo.intramural_threshold_pct",
        "uncertainty.n_radius_draws",
        "uncertainty.n_mask_draws",
        "uncertainty.through_plane_frac",
        "body.adaptive_radius_scale",
        "fibroid.contact_distance_mm",
        "figo.type3_max_intracavitary_pct",
        "borderline.band_pct",
        "robustness.decimation_factors",
    ],
)
def test_required_keys_present(cfg, key):
    assert cfg[key] is not None


def test_label_scheme_is_the_umd_scheme(cfg):
    labels = {cfg[f"labels.{n}"] for n in
              ("background", "wall", "cavity", "fibroid", "nabothian")}
    assert labels == {0, 1, 2, 3, 4}


def test_mask_glob_covers_the_seq_typo(cfg):
    """F1: 33 masks are named _seq, not _seg. A _seg-only glob drops them."""
    import fnmatch
    pattern = cfg["data.mask_glob"]
    assert fnmatch.fnmatch("UMD_221129_001_seg.nii.gz", pattern)
    assert fnmatch.fnmatch("UMD_221129_045_seq.nii.gz", pattern)


def test_dicom_spacing_outranks_nifti(cfg):
    """F2: NIfTI pixdim is (1,1,1) for the 33 raw files and must never win."""
    priority = cfg["spacing.source_priority"]
    assert priority.index("dicom") < priority.index("nifti")
    assert cfg["spacing.reject_unit_isotropic"] is True


def test_slice_bounds_reject_the_unset_pixdim(cfg):
    """A 1.0 mm slice must fall outside the accepted band, or F2 slips through."""
    assert not (cfg["spacing.min_slice_mm"] <= 1.0 <= cfg["spacing.max_slice_mm"])
