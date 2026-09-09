"""Phase 0 guard: the dataset copy is intact. Skipped when data/ is absent."""

import pytest

from figomeas import load_config
from figomeas.config import REPO_ROOT

cfg = load_config()
ROOT = REPO_ROOT / cfg["data.root"]

pytestmark = pytest.mark.skipif(not ROOT.exists(), reason="UMD dataset not present")


def test_three_hundred_patients():
    assert len(sorted(p for p in ROOT.iterdir() if p.is_dir())) == 300


def test_mask_glob_finds_every_patient():
    """The whole point of F1: 267 _seg + 33 _seq = 300, no silent drops."""
    masks = sorted(ROOT.glob(f"*/{cfg['data.mask_glob']}"))
    assert len(masks) == 300
    assert sum(1 for m in masks if "_seg" in m.name) == 267
    assert sum(1 for m in masks if "_seq" in m.name) == 33


def test_dicom_available_for_spacing_recovery():
    """F2 is only fixable because DICOM ships for all 300 patients."""
    with_dcm = [p for p in ROOT.iterdir() if p.is_dir() and any(p.glob("*.dcm"))]
    assert len(with_dcm) == 300
