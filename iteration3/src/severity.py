
from __future__ import annotations

from config import CATEGORIES, POTENCY_WEIGHTS


def severity(categories: dict, weights: dict | None = None) -> float:
    """Collapse per-category presence scores into one severity value in [0, 1].

    Noisy-OR over potency-weighted scores:

        severity = 1 - prod_c (1 - w_c * s_c)

    i.e. the probability that at least one manipulation lands, where each
    pattern lands with probability (presence score x potency weight). One
    potent pattern alone yields high severity; stacked patterns compound with
    diminishing returns. is_dark is not an input — System A's category scores
    already embed it, so including it would double-count.
    """
    if weights is None:
        weights = POTENCY_WEIGHTS

    prod = 1.0
    for cat in CATEGORIES:
        if cat not in categories:
            raise KeyError(f"missing category {cat!r} (got {sorted(categories)})")
        s = float(categories[cat])
        if not 0.0 <= s <= 1.0:
            raise ValueError(f"{cat} score {s} outside [0, 1]")
        w = float(weights[cat])
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"{cat} weight {w} outside [0, 1]")
        prod *= 1.0 - w * s
    return 1.0 - prod
