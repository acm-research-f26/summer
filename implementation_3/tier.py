"""
tier.py — impact-tier assignment logic for MIRA implementation_3.

The tier is derived deterministically from square footage and estimated
megawatt draw. This module keeps that business logic isolated so it can
be unit-tested independently of the notebook.
"""

from config import SQFT_COLO_MIN, SQFT_HYPER_MIN, MW_COLO_MIN, MW_HYPER_MIN


def assign_impact_tier(sqft: int, est_mw: float) -> int:
    """Return the impact tier (0 / 1 / 2) for a building.

    Parameters
    ----------
    sqft:
        Gross floor area in square feet.
    est_mw:
        Estimated power draw in megawatts (typically sqft * 150 W/sqft / 1e6).

    Returns
    -------
    int
        0 = Edge/Enterprise, 1 = Colocation, 2 = Hyperscale.
    """
    if sqft >= SQFT_HYPER_MIN or est_mw >= MW_HYPER_MIN:
        return 2
    if sqft >= SQFT_COLO_MIN or est_mw >= MW_COLO_MIN:
        return 1
    return 0
