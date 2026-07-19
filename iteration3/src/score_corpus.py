
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    AGGREGATE_FILES,
    CATEGORIES,
    EXPECTED_SCREENS,
    IT2_CACHE_VLM,
    IT2_PROMPT_VERSION,
    IT2_ROOT,
    MAX_MISSING_VISION,
    PATHS,
    POTENCY_WEIGHTS,
    POTENCY_WEIGHTS_ALT,
    ROOT,
    VLM_CACHE_SUFFIX,
)
from src.severity import severity

DATASETS = ("contextdp", "own")

SEVERITY_COLUMNS = [
    "severity_text",
    "severity_text_alt",
    "severity_vision",
    "severity_vision_alt",
]

CORPUS_COLUMNS = (
    ["screen_id", "source_dataset", "platform", "path"]
    + [f"gt_{c}" for c in CATEGORIES]
    + ["gt_is_dark"]
    + [f"text_{c}" for c in CATEGORIES]
    + ["text_is_dark"]
    + [f"vision_{c}" for c in CATEGORIES]
    + ["vision_is_dark", "vision_parse_failed"]
    + SEVERITY_COLUMNS
)


# --- Ground truth via iteration2 ---------------------------------------------

def load_truth(force: bool = False) -> dict[str, pd.DataFrame]:
    """Truth dataframes per dataset, loaded once via it2_worker and cached."""
    cache = {ds: PATHS["data"] / f"truth_{ds}.csv" for ds in DATASETS}
    if not force and all(p.exists() for p in cache.values()):
        return {ds: pd.read_csv(p) for ds, p in cache.items()}

    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "truth.json"
        subprocess.run(
            [sys.executable, str(ROOT / "src/it2_worker.py"), str(out)],
            cwd=IT2_ROOT,
            check=True,
        )
        payload = json.loads(out.read_text())

    truth = {}
    for ds in DATASETS:
        df = pd.DataFrame(payload[ds])
        cache[ds].parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache[ds], index=False)
        truth[ds] = df
    return truth


# --- Cached predictions ------------------------------------------------------

def _reconstruct_vision(truth: pd.DataFrame, dataset: str) -> dict:
    """Rebuild a missing system_b aggregate from per-image VLM cache files.

    Read-only: hashes each image and looks up its cached record; screens with
    no cached record are skipped and surface as NaN rows. Never calls the VLM.
    """
    out = {}
    for row in truth.itertuples():
        img_hash = hashlib.sha256(Path(row.path).read_bytes()).hexdigest()
        cache_file = IT2_CACHE_VLM / f"{img_hash}{VLM_CACHE_SUFFIX}"
        if cache_file.exists():
            out[row.filename] = json.loads(cache_file.read_text())

    rebuilt = PATHS["data"] / f"system_b_{dataset}_reconstructed.json"
    rebuilt.parent.mkdir(parents=True, exist_ok=True)
    rebuilt.write_text(json.dumps(out, indent=1))
    print(
        f"[corpus] reconstructed vision aggregate for {dataset}: "
        f"{len(out)}/{len(truth)} screens from per-image cache -> {rebuilt}"
    )
    return out


def load_aggregates(truth: dict[str, pd.DataFrame]) -> dict[tuple, dict]:
    """The four screen-level prediction dicts, keyed (modality, dataset)."""
    preds = {}
    for key, path in AGGREGATE_FILES.items():
        modality, dataset = key
        if path.exists():
            preds[key] = json.loads(path.read_text())
        elif modality == "vision":
            print(f"[corpus] {path} missing — reconstructing from per-image cache")
            preds[key] = _reconstruct_vision(truth[dataset], dataset)
        else:
            raise FileNotFoundError(
                f"required text aggregate missing: {path} — iteration 2's "
                f"cache is incomplete; do NOT re-run OCR from iteration 3"
            )
    return preds


# --- Table construction ------------------------------------------------------

def build_corpus_table(
    truth: dict[str, pd.DataFrame],
    preds_text: dict[str, dict],
    preds_vision: dict[str, dict],
) -> pd.DataFrame:
    rows = []
    for ds in truth:
        for t in truth[ds].itertuples():
            text = preds_text[ds].get(t.filename)
            if text is None:
                raise KeyError(f"no text prediction for {ds}/{t.filename}")
            vision = preds_vision[ds].get(t.filename)

            row = {
                "screen_id": t.filename,
                "source_dataset": ds,
                "platform": t.platform,
                "path": t.path,
                "gt_is_dark": int(t.is_dark),
                "text_is_dark": text["is_dark"],
                "vision_is_dark": vision["is_dark"] if vision else np.nan,
                "vision_parse_failed": bool(vision.get("parse_failed", False))
                if vision
                else np.nan,
            }
            for c in CATEGORIES:
                row[f"gt_{c}"] = int(getattr(t, c))
                row[f"text_{c}"] = text["categories"][c]
                row[f"vision_{c}"] = vision["categories"][c] if vision else np.nan

            row["severity_text"] = severity(text["categories"], POTENCY_WEIGHTS)
            row["severity_text_alt"] = severity(text["categories"], POTENCY_WEIGHTS_ALT)
            if vision:
                row["severity_vision"] = severity(vision["categories"], POTENCY_WEIGHTS)
                row["severity_vision_alt"] = severity(
                    vision["categories"], POTENCY_WEIGHTS_ALT
                )
            else:
                row["severity_vision"] = np.nan
                row["severity_vision_alt"] = np.nan
            rows.append(row)

    return pd.DataFrame(rows)[CORPUS_COLUMNS]


# --- Driver ------------------------------------------------------------------

def main() -> pd.DataFrame:
    truth = load_truth()
    for ds, df in truth.items():
        assert len(df) == EXPECTED_SCREENS[ds], (
            f"{ds}: {len(df)} truth rows, expected {EXPECTED_SCREENS[ds]}"
        )

    preds = load_aggregates(truth)
    corpus = build_corpus_table(
        truth,
        {ds: preds[("text", ds)] for ds in DATASETS},
        {ds: preds[("vision", ds)] for ds in DATASETS},
    )

    n = len(corpus)
    assert n == sum(EXPECTED_SCREENS.values()), f"{n} corpus rows"
    text_cols = [f"text_{c}" for c in CATEGORIES] + ["text_is_dark", "severity_text"]
    assert not corpus[text_cols].isna().any().any(), "NaN in text columns"
    n_missing_vision = int(corpus["severity_vision"].isna().sum())
    assert n_missing_vision <= MAX_MISSING_VISION, (
        f"{n_missing_vision} screens missing vision predictions"
    )
    for col in SEVERITY_COLUMNS:
        vals = corpus[col].dropna()
        assert ((vals >= 0) & (vals <= 1)).all(), f"{col} out of bounds"

    PATHS["data"].mkdir(parents=True, exist_ok=True)
    PATHS["reports"].mkdir(parents=True, exist_ok=True)
    corpus.to_csv(PATHS["corpus_scores"], index=False)
    corpus.to_csv(PATHS["reports"] / "corpus_scores.csv", index=False)

    print(f"[corpus] {n} screens -> {PATHS['corpus_scores']}")
    print(f"[corpus] vision scores from prompt {IT2_PROMPT_VERSION}")
    if n_missing_vision:
        missing = corpus.loc[corpus["severity_vision"].isna(), "screen_id"].tolist()
        print(
            f"[corpus] {n_missing_vision} screens have NO cached vision "
            f"prediction (excluded from vision-side analyses): {missing}"
        )
    print("[corpus] severity distributions:")
    print(corpus[SEVERITY_COLUMNS].describe().round(3).to_string())
    return corpus


if __name__ == "__main__":
    main()
