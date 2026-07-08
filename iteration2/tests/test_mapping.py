"""Every CONTEXTDP label must map somewhere — no silent drops."""
import pytest

from config import CATEGORIES, CONTEXTDP_ROOT, LABEL_MAPPING


def test_mapping_targets_are_valid():
    for name, target in LABEL_MAPPING.items():
        assert target is None or target in CATEGORIES, (name, target)


def test_no_dp_maps_to_none():
    assert LABEL_MAPPING["NO DP"] is None


@pytest.mark.skipif(
    not CONTEXTDP_ROOT.exists(),
    reason="CONTEXTDP not present (set CONTEXTDP_ROOT or clone AidUI)",
)
def test_every_dataset_category_is_mapped():
    from src.dataset_contextdp import contextdp_category_names

    names = contextdp_category_names()
    unmapped = names - set(LABEL_MAPPING)
    assert not unmapped, f"CONTEXTDP categories with no mapping: {unmapped}"


@pytest.mark.skipif(
    not CONTEXTDP_ROOT.exists(),
    reason="CONTEXTDP not present (set CONTEXTDP_ROOT or clone AidUI)",
)
def test_screen_labels_match_known_totals():
    from src.dataset_contextdp import load_contextdp

    df = load_contextdp()
    assert len(df) == 501
    assert int((df["is_dark"] == 0).sum()) == 243  # non-DP count per AidUI README
    assert int(df["guilt_wording"].sum()) == 0     # no confirmshaming label exists
    # mobile contributes no text-flavored categories (dataset property)
    mobile = df[df["platform"] == "mobile"]
    assert int(mobile[["urgency", "scarcity", "social_proof"]].to_numpy().sum()) == 0
