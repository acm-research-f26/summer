"""
config.py — shared constants and column definitions for MIRA implementation_3.

Import this module at the top of any script or notebook cell to get a single
source of truth for all feature lists, tier thresholds, and training constants.
"""

# ---------------------------------------------------------------------------
# Power / size thresholds used to assign impact tiers
# ---------------------------------------------------------------------------
MW_DENSITY_W_PER_SQFT = 150
MW_DIVISOR = 1_000_000

SQFT_COLO_MIN = 50_000
SQFT_HYPER_MIN = 200_000
MW_COLO_MIN = 5.0
MW_HYPER_MIN = 40.0

# Human-readable labels for the three impact tiers (0 / 1 / 2)
TIER_LABELS: dict[int, str] = {
    0: "Edge/Enterprise",
    1: "Colocation",
    2: "Hyperscale",
}

# ---------------------------------------------------------------------------
# WRI Aqueduct source column names
# ---------------------------------------------------------------------------
SOURCE_METADATA_COLUMNS = [
    "points_id",
    "location_name",
    "longitude",
    "latitude",
]

# All 13 WRI indicator score columns in the Aqueduct response
WRI_SCORE_COLUMNS_ALL = [
    "bws_score",
    "bwd_score",
    "iav_score",
    "sev_score",
    "gtd_score",
    "rfr_score",
    "cfr_score",
    "drr_score",
    "ucw_score",
    "cep_score",
    "udw_score",
    "usa_score",
    "rri_score",
]

# Columns dropped from WRI representation due to high sentinel rate,
# zero variance, or negligible variance.
DROP_COLUMNS = ["gtd_score", "ucw_score", "rri_score", "usa_score"]

# Nine-feature WRI representation used for modeling
WRI_SCORE_COLUMNS = [
    col for col in WRI_SCORE_COLUMNS_ALL if col not in DROP_COLUMNS
]

# Four Electric-Power-weighted aggregate scores
ELP_SCORE_COLUMNS = [
    "w_awr_elp_qan_score",  # Water Quantity risk
    "w_awr_elp_qal_score",  # Water Quality risk
    "w_awr_elp_rrr_score",  # Regulatory & Reputational risk
    "w_awr_elp_tot_score",  # Total (overall weighted) risk
]

# Feature sets used in the WRI-vs-ELP comparison
FEATURE_COLUMNS_BY_SET: dict[str, list[str]] = {
    "WRI indicators": WRI_SCORE_COLUMNS,
    "ELP aggregates": ELP_SCORE_COLUMNS,
}

# ---------------------------------------------------------------------------
# Modeling metadata and target
# ---------------------------------------------------------------------------
METADATA_COLUMNS = ["id", "name", "lon", "lat", "impact_tier"]
TARGET_COLUMN = "impact_tier"

# Sentinel value used by WRI Aqueduct for missing / unpublished data
WRI_SENTINEL = -9999.0

# ---------------------------------------------------------------------------
# Shared training constants
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
