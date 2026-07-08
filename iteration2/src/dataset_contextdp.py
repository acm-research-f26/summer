
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config import (
    CATEGORIES,
    CONTEXTDP_ROOT,
    DEV_SUBSET_SEED,
    DEV_SUBSET_SIZE,
    LABEL_MAPPING,
)

PLATFORMS = ("web", "mobile")


def load_result_json(platform_dir: Path) -> dict:
    with open(platform_dir / "result.json") as f:
        return json.load(f)


def contextdp_category_names(root: Path = CONTEXTDP_ROOT) -> set[str]:
    """Every category name that appears in either platform's result.json."""
    names: set[str] = set()
    for plat in PLATFORMS:
        data = load_result_json(root / plat)
        names.update(c["name"] for c in data["categories"])
    return names


def load_contextdp(root: Path = CONTEXTDP_ROOT) -> pd.DataFrame:
    """One row per screenshot: path, platform, 5 category flags, is_dark.

    Raises KeyError on any CONTEXTDP category name missing from
    LABEL_MAPPING — unmapped labels must never be silently dropped.
    """
    rows = []
    for plat in PLATFORMS:
        plat_dir = root / plat
        data = load_result_json(plat_dir)
        id2name = {c["id"]: c["name"] for c in data["categories"]}

        flags_by_image: dict[int, set[str]] = {img["id"]: set() for img in data["images"]}
        for ann in data["annotations"]:
            name = id2name[ann["category_id"]]
            if name not in LABEL_MAPPING:
                raise KeyError(
                    f"CONTEXTDP category {name!r} has no entry in config.LABEL_MAPPING"
                )
            mapped = LABEL_MAPPING[name]
            if mapped is not None:
                flags_by_image[ann["image_id"]].add(mapped)

        for img in data["images"]:
            flags = flags_by_image[img["id"]]
            row = {
                "filename": Path(img["file_name"]).name,
                "path": str(plat_dir / "images" / Path(img["file_name"]).name),
                "platform": plat,
            }
            for cat in CATEGORIES:
                row[cat] = int(cat in flags)
            row["is_dark"] = int(bool(flags))
            rows.append(row)

    df = pd.DataFrame(rows)
    missing = [p for p in df["path"] if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} images referenced by result.json are missing, "
            f"e.g. {missing[:3]}"
        )
    return df


def select_dev_subset(
    df: pd.DataFrame,
    size: int = DEV_SUBSET_SIZE,
    seed: int = DEV_SUBSET_SEED,
) -> pd.DataFrame:
    """Fixed dev subset for prompt tuning: stratified over platform x is_dark.

    Deterministic (seeded); the chosen filenames are written to
    reports/dev_subset.txt by run_comparison so the tuning set is on record.
    """
    strata = df.groupby(["platform", "is_dark"], group_keys=False)
    # proportional allocation, at least 1 per stratum
    picked = strata.apply(
        lambda g: g.sample(
            n=max(1, round(size * len(g) / len(df))), random_state=seed
        ),
        include_groups=False,
    )
    # trim/pad to exactly `size` deterministically
    picked = picked.sample(frac=1, random_state=seed).head(size)
    return df.loc[sorted(picked.index)]


def label_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Sanity table: positives per category, per platform and overall."""
    parts = []
    for plat, g in df.groupby("platform"):
        parts.append({"platform": plat, "n_screens": len(g),
                      **{c: int(g[c].sum()) for c in CATEGORIES},
                      "is_dark": int(g["is_dark"].sum())})
    parts.append({"platform": "all", "n_screens": len(df),
                  **{c: int(df[c].sum()) for c in CATEGORIES},
                  "is_dark": int(df["is_dark"].sum())})
    return pd.DataFrame(parts)


if __name__ == "__main__":
    df = load_contextdp()
    print(f"Loaded {len(df)} screens from {CONTEXTDP_ROOT}")
    print(label_distribution(df).to_string(index=False))
