"""Build deduped train/val/test parquets + label_maps.json for both tasks.

Run:  python -m src.data_prep

Mathur positives (multi-class) and dataset.tsv (binary) are deduped IDENTICALLY
(strip whitespace, drop empty, drop duplicates on the text column) so that the
binary task does not leak. Counts of dropped rows are logged. Confirmshaming
takes precedence over its Misdirection parent before the Other fallback.
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    BINARY_LABELS,
    MULTICLASS_LABELS,
    PATHS,
    SEED,
    TEST_FRAC,
    VAL_FRAC,
)

MATHUR_TEXT_COL = "Pattern String"
CONFIRMSHAMING = "Confirmshaming"


def _load_augmentation() -> pd.DataFrame | None:
    """Hand-written augmentation rows (schema: text,is_dark,category).

    Adds fluent benign negatives (to break the page-chrome register shortcut in
    the binary task) and plainly-worded positives (weighted toward the weakest
    classes, urgency/guilt_wording). Returns None when the file is absent so the
    pipeline still runs unaugmented.
    """
    path = PATHS["augmentation"]
    if not path.exists():
        print(f"  (no augmentation file at {path} — skipping)")
        return None
    aug = pd.read_csv(path)
    aug["text"] = aug["text"].astype("string").str.strip()
    aug["category"] = aug["category"].astype("string").str.strip()
    aug["is_dark"] = aug["is_dark"].astype(int)
    aug = aug[aug["text"].notna() & (aug["text"] != "")].drop_duplicates("text")
    print(f"  loaded {len(aug)} augmentation rows from {path.name}")
    return aug.reset_index(drop=True)


def _augment_train(
    splits: dict[str, pd.DataFrame],
    aug_rows: pd.DataFrame,
    name: str,
) -> dict[str, pd.DataFrame]:
    """Append augmentation rows to TRAIN only.

    val/test are left untouched so they stay comparable to the unaugmented run,
    and any augmentation text already present in any split is dropped to avoid
    leakage / duplication.
    """
    if aug_rows is None or aug_rows.empty:
        return splits
    existing = set(
        pd.concat([splits[s]["text"] for s in ("train", "val", "test")]).astype(str)
    )
    fresh = aug_rows[~aug_rows["text"].astype(str).isin(existing)]
    dropped = len(aug_rows) - len(fresh)
    add = fresh[["text", "label"]].reset_index(drop=True)
    splits["train"] = pd.concat([splits["train"], add], ignore_index=True)
    print(f"  {name} augmentation: +{len(add)} train rows "
          f"(dropped {dropped} already-present), train now {len(splits['train'])}")
    return splits


def _to_5class_key(pattern_type: str, pattern_category: str) -> str:
    if pattern_type == CONFIRMSHAMING:
        return "guilt_wording"
    if pattern_category == "Urgency":
        return "urgency"
    if pattern_category == "Scarcity":
        return "scarcity"
    if pattern_category == "Social Proof":
        return "social_proof"
    return "other"


def _strip_and_dedupe(df: pd.DataFrame, col: str, name: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].astype("string").str.strip()
    empty = df[col].isna() | (df[col] == "")
    print(f"  {name}: empty {col!r} dropped = {int(empty.sum())}")
    df = df[~empty]
    before = len(df)
    df = df.drop_duplicates(subset=[col])
    print(f"  {name}: duplicate {col!r} dropped = {before - len(df)}")
    print(f"  {name}: rows after dedup = {len(df)}")
    return df.reset_index(drop=True)


def _stratified_split(df: pd.DataFrame, name: str) -> dict[str, pd.DataFrame]:
    """70/15/15 stratified split with random_state=SEED, applied via two calls."""
    y = df["label"]
    rest_df, test_df, rest_y, _ = train_test_split(
        df, y, test_size=TEST_FRAC, random_state=SEED, stratify=y,
    )
    val_relative = VAL_FRAC / (1.0 - TEST_FRAC)
    train_df, val_df = train_test_split(
        rest_df, test_size=val_relative, random_state=SEED, stratify=rest_y,
    )
    print(f"  {name} split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    return {
        "train": train_df.reset_index(drop=True),
        "val":   val_df.reset_index(drop=True),
        "test":  test_df.reset_index(drop=True),
    }


def build_multiclass() -> pd.DataFrame:
    print("== Mathur (multi-class) ==")
    m = pd.read_csv(PATHS["raw_mathur"])
    print(f"  raw rows = {len(m)}")
    print("  raw Pattern Category counts:")
    for cat, n in m["Pattern Category"].value_counts(dropna=False).items():
        print(f"    {cat}: {n}")

    m = _strip_and_dedupe(m, MATHUR_TEXT_COL, "Mathur")
    m["Pattern Type"] = m["Pattern Type"].astype("string").str.strip()
    m["Pattern Category"] = m["Pattern Category"].astype("string").str.strip()
    m["label_key"] = [
        _to_5class_key(pt, pc) for pt, pc in zip(m["Pattern Type"], m["Pattern Category"])
    ]

    key_to_id = {k: i for i, k, _ in MULTICLASS_LABELS}
    expected_keys = set(key_to_id)
    found_keys = set(m["label_key"].unique())
    assert found_keys <= expected_keys, f"unexpected label keys: {found_keys - expected_keys}"

    print("  final 5-class counts (post-dedup):")
    for i, key, display in MULTICLASS_LABELS:
        n = int((m["label_key"] == key).sum())
        print(f"    {display:14s} ({key:14s}): {n}")
    total = sum(int((m["label_key"] == k).sum()) for _, k, _ in MULTICLASS_LABELS)
    assert total == len(m), "5-class assignment lost rows"
    print(f"    TOTAL: {total}")

    return pd.DataFrame({
        "text":  m[MATHUR_TEXT_COL].astype(str).values,
        "label": m["label_key"].map(key_to_id).astype(int).values,
    })


def build_binary() -> pd.DataFrame:
    print("== dataset.tsv (binary) ==")
    d = pd.read_csv(PATHS["raw_dataset_tsv"], sep="\t")
    print(f"  raw rows = {len(d)}")
    print(f"  raw label counts = {d['label'].value_counts().to_dict()}")
    d = _strip_and_dedupe(d, "text", "dataset.tsv")
    print(f"  binary balance post-dedup = {d['label'].value_counts().to_dict()}")
    return pd.DataFrame({
        "text":  d["text"].astype(str).values,
        "label": d["label"].astype(int).values,
    })


def write_label_maps() -> None:
    label_maps = {
        "binary": {str(i): name for i, name in BINARY_LABELS.items()},
        "multiclass": {
            str(i): {"key": k, "display": d} for i, k, d in MULTICLASS_LABELS
        },
    }
    PATHS["processed"].mkdir(parents=True, exist_ok=True)
    with open(PATHS["label_maps"], "w") as f:
        json.dump(label_maps, f, indent=2)
    print(f"Wrote label maps -> {PATHS['label_maps']}")


def main() -> None:
    PATHS["processed"].mkdir(parents=True, exist_ok=True)
    write_label_maps()

    print("== augmentation ==")
    aug = _load_augmentation()
    key_to_id = {k: i for i, k, _ in MULTICLASS_LABELS}

    multi = build_multiclass()
    multi_splits = _stratified_split(multi, "multiclass")
    if aug is not None:
        # multi-class trains on dark rows only; map category -> id.
        mc_aug = aug[aug["is_dark"] == 1].copy()
        bad = set(mc_aug["category"]) - set(key_to_id)
        assert not bad, f"augmentation has unknown categories: {bad}"
        mc_aug["label"] = mc_aug["category"].map(key_to_id).astype(int)
        multi_splits = _augment_train(multi_splits, mc_aug, "multiclass")
    for name, df in multi_splits.items():
        df.to_parquet(PATHS["processed"] / f"multiclass_{name}.parquet", index=False)

    binary = build_binary()
    binary_splits = _stratified_split(binary, "binary")
    if aug is not None:
        bin_aug = aug.copy()
        bin_aug["label"] = bin_aug["is_dark"].astype(int)
        binary_splits = _augment_train(binary_splits, bin_aug, "binary")
    for name, df in binary_splits.items():
        df.to_parquet(PATHS["processed"] / f"binary_{name}.parquet", index=False)

    print("data_prep done.")


if __name__ == "__main__":
    main()
