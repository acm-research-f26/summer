
from __future__ import annotations

from config import CATEGORIES


def empty_screen_result() -> dict:
    """Screen with no usable text: nothing detected (the honest
    'text can't see it' case for the baseline)."""
    return {
        "is_dark": 0.0,
        "categories": {c: 0.0 for c in CATEGORIES},
        "top_category": "other",
    }


def aggregate_snippets(snippet_preds: list[dict]) -> dict:
    """Aggregate iteration-1 per-snippet outputs to one screen-level result.

    `snippet_preds` are frozen-contract dicts from it1's predict_batch.
    Returns the screen-level shape documented above.
    """
    if not snippet_preds:
        return empty_screen_result()

    cats = {
        c: max(p["is_dark"] * p["categories"][c] for p in snippet_preds)
        for c in CATEGORIES
    }
    return {
        "is_dark": max(p["is_dark"] for p in snippet_preds),
        "categories": cats,
        "top_category": max(cats, key=cats.get),
    }
