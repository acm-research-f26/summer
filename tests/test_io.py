"""Phase 1: the loader must defeat the container and spacing traps by construction."""

import gzip

import nibabel as nib
import numpy as np
import pytest

from figomeas import load_config
from figomeas.io import DicomGeometry, load_mask, resolve_spacing


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture
def volume():
    rng = np.random.default_rng(0)
    return rng.integers(0, 5, size=(8, 9, 4), dtype=np.uint8)


def _write(tmp_path, volume, name, spacing, compress):
    """Write a NIfTI, optionally gzipped, under a deliberately misleading name."""
    img = nib.Nifti1Image(volume, np.diag([*spacing, 1.0]))
    img.header.set_zooms(spacing)
    payload = img.to_bytes()
    path = tmp_path / name
    path.write_bytes(gzip.compress(payload) if compress else payload)
    return path


def test_gzip_container_loads(tmp_path, volume):
    p = _write(tmp_path, volume, "UMD_x_seg.nii.gz", (0.5, 0.5, 5.0), compress=True)
    m = load_mask(p)
    assert m.container == "gzip"
    assert m.filename_kind == "seg"
    np.testing.assert_array_equal(m.data, volume)


def test_raw_nifti_wearing_a_gz_extension_still_loads(tmp_path, volume):
    """F1: 34 files are raw despite `.gz`. Trusting the extension loses them."""
    p = _write(tmp_path, volume, "UMD_x_seq.nii.gz", (0.5, 0.5, 5.0), compress=False)
    m = load_mask(p)
    assert m.container == "raw"
    assert m.filename_kind == "seq"
    np.testing.assert_array_equal(m.data, volume)


def test_container_is_sniffed_not_inferred_from_name(tmp_path, volume):
    """F3: UMD_221129_037 is named _seg but is raw. Name must not decide container."""
    p = _write(tmp_path, volume, "UMD_2221129_037_seg.nii.gz", (0.5, 0.5, 5.0), compress=False)
    m = load_mask(p)
    assert m.container == "raw"
    assert m.filename_kind == "seg"


def test_both_containers_give_identical_arrays(tmp_path, volume):
    a = load_mask(_write(tmp_path, volume, "a_seg.nii.gz", (0.5, 0.5, 5.0), True))
    b = load_mask(_write(tmp_path, volume, "b_seq.nii.gz", (0.5, 0.5, 5.0), False))
    np.testing.assert_array_equal(a.data, b.data)


# ----------------------------------------------------------------- spacing policy


def _geom(spacing):
    return DicomGeometry(spacing, 20, 512, 512, "ImagePositionPatient", "2D", "T2W", "uid")


def test_dicom_wins_over_nifti(cfg):
    spacing, source = resolve_spacing((0.446, 0.446, 6.6), _geom((0.5, 0.5, 5.0)), cfg)
    assert source == "dicom"
    assert spacing == (0.5, 0.5, 5.0)


def test_unset_unit_pixdim_is_rejected(cfg):
    """F2: (1,1,1) is an export artifact. It must never be accepted as real."""
    spacing, source = resolve_spacing((1.0, 1.0, 1.0), _geom((0.446, 0.446, 5.0)), cfg)
    assert source == "dicom"
    assert spacing == (0.446, 0.446, 5.0)


def test_unit_pixdim_is_rejected_even_with_no_dicom(cfg):
    """Better to resolve nothing than to silently accept a 1x anisotropy."""
    spacing, source = resolve_spacing((1.0, 1.0, 1.0), _geom(None), cfg)
    assert source == "none"
    assert spacing is None


def test_nifti_used_when_dicom_absent(cfg):
    spacing, source = resolve_spacing((0.446, 0.446, 6.6), _geom(None), cfg)
    assert source == "nifti"
    assert spacing == (0.446, 0.446, 6.6)


@pytest.mark.parametrize("bad", [(0.05, 0.05, 5.0), (5.0, 5.0, 5.0), (0.5, 0.5, 0.4),
                                 (0.5, 0.5, 40.0), (0.5, 0.5, 0.0)])
def test_implausible_spacings_are_refused(cfg, bad):
    spacing, source = resolve_spacing(None, _geom(bad), cfg)
    assert source == "none"
