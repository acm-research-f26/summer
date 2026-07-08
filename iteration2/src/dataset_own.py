
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import CATEGORIES, OWN_SCREENSHOTS_ROOT

EXPECTED_COLUMNS = ["filename", *CATEGORIES]


def load_own(root: Path = OWN_SCREENSHOTS_ROOT) -> pd.DataFrame:
    labels_csv = root / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"expected {labels_csv}")

    df = pd.read_csv(labels_csv)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            f"labels.csv columns {list(df.columns)} != expected {EXPECTED_COLUMNS}"
        )
    if df.empty:
        raise ValueError(f"{labels_csv} has no rows yet — add screenshots first")

    bad = df[~df[CATEGORIES].isin([0, 1]).all(axis=1)]
    if not bad.empty:
        raise ValueError(f"non-0/1 label values in rows: {bad['filename'].tolist()}")

    df = df.copy()
    df["path"] = df["filename"].map(lambda f: str(root / f))
    df["platform"] = "own"
    df["is_dark"] = df[CATEGORIES].max(axis=1).astype(int)

    missing = [p for p in df["path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} files listed in labels.csv are missing: {missing[:5]}"
        )
    return df[["filename", "path", "platform", *CATEGORIES, "is_dark"]]
