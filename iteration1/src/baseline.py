"""TF-IDF (1-2 gram) + LogisticRegression baseline for both tasks.

Run:  python -m src.baseline

Trains on the 70% train split, picks C from BASELINE_C_GRID by val macro-F1,
and KEEPS the train-only fit (no refit on train+val) so the comparison with
DistilBERT is apples-to-apples.
"""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from config import BASELINE_C_GRID, PATHS, SEED


def _make_pipeline(C: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True,
        )),
        ("clf", LogisticRegression(
            class_weight="balanced", max_iter=2000, C=C, random_state=SEED,
            n_jobs=None,
        )),
    ])


def train_task(task: str) -> None:
    print(f"== Baseline: {task} ==")
    train = pd.read_parquet(PATHS["processed"] / f"{task}_train.parquet")
    val = pd.read_parquet(PATHS["processed"] / f"{task}_val.parquet")

    best_C: float | None = None
    best_f1 = -1.0
    best_pipe: Pipeline | None = None
    for C in BASELINE_C_GRID:
        pipe = _make_pipeline(C)
        pipe.fit(train["text"], train["label"])
        pred = pipe.predict(val["text"])
        f1 = f1_score(val["label"], pred, average="macro")
        print(f"  C={C:>5}: val macro-F1 = {f1:.4f}")
        if f1 > best_f1:
            best_f1, best_C, best_pipe = f1, C, pipe
    assert best_pipe is not None
    print(f"  -> picked C={best_C}, val macro-F1={best_f1:.4f} (no train+val refit)")

    PATHS["models"].mkdir(parents=True, exist_ok=True)
    out = PATHS["models"] / f"baseline_{task}.joblib"
    joblib.dump(best_pipe, out)
    print(f"  wrote {out}")


def main() -> None:
    for task in ("binary", "multiclass"):
        train_task(task)


if __name__ == "__main__":
    main()
