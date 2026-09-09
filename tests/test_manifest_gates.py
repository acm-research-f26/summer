"""Phase 1 exit gates, re-asserted as tests against the built manifest."""

import pandas as pd
import pytest

from figomeas import load_config
from figomeas.config import REPO_ROOT
from figomeas.manifest import MANIFEST_COLUMNS, check_gates

MANIFEST = REPO_ROOT / "results" / "manifest.csv"
pytestmark = pytest.mark.skipif(not MANIFEST.exists(),
                                reason="manifest not built (run `make manifest`)")


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(MANIFEST)


def test_schema(df):
    assert list(df.columns) == MANIFEST_COLUMNS


def test_all_phase1_gates_pass(df):
    failed = [(n, d) for n, ok, d in check_gates(df, load_config()) if not ok]
    assert not failed, f"failed gates: {failed}"


def test_every_patient_resolved_from_dicom(df):
    assert (df.spacing_source == "dicom").all()


def test_no_patient_kept_the_unset_pixdim(df):
    kept = df[(df.used_sx == 1.0) & (df.used_sy == 1.0) & (df.used_sz == 1.0)]
    assert kept.empty


def test_seq_patients_were_actually_corrected(df):
    """Their NIfTI said (1,1,1); their used spacing must not."""
    seq = df[df.filename_kind == "seq"]
    assert len(seq) == 33
    assert (seq.nifti_sz == 1.0).all()
    assert (seq.used_sz > 3.0).all()


def test_fibroid_component_count_is_plausible(df):
    """~1,077 fibroids documented; pre-filter count should be in the same ballpark."""
    assert 900 <= int(df.n_fibroid_components.sum()) <= 1400
