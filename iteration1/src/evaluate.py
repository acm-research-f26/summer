"""Evaluate baseline and DistilBERT (binary + multi-class) on the test split.

Run:  python -m src.evaluate

Writes for each available {task x model}:
  reports/confusion_matrices/{task}_{model}.csv    raw counts, labeled
  reports/confusion_matrices/{task}_{model}.png    seaborn heatmap
  reports/comparison.csv                           accuracy/macro_f1/weighted_f1
  reports/comparison.md                            same table + validity caveats

The two validity caveats (register-contaminated is_dark, Social Proof template
memorization) are emitted verbatim into comparison.md every run.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from config import BINARY_LABELS, MAX_LENGTH, MULTICLASS_LABELS, PATHS


def _labels_for(task: str) -> tuple[list[int], list[str]]:
    if task == "binary":
        return [0, 1], [BINARY_LABELS[0], BINARY_LABELS[1]]
    return [i for i, _, _ in MULTICLASS_LABELS], [d for _, _, d in MULTICLASS_LABELS]


def _save_confusion(y_true, y_pred, task: str, model: str,
                    ids: list[int], names: list[str]) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=ids)
    df = pd.DataFrame(cm, index=names, columns=names)
    PATHS["confusion_matrices"].mkdir(parents=True, exist_ok=True)
    df.to_csv(PATHS["confusion_matrices"] / f"{task}_{model}.csv")
    width = max(4, int(len(names) * 1.1))
    plt.figure(figsize=(width, max(3, int(len(names) * 0.8))))
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{task} — {model}")
    plt.tight_layout()
    plt.savefig(PATHS["confusion_matrices"] / f"{task}_{model}.png", dpi=120)
    plt.close()


def _summary_row(y_true, y_pred, task: str, model: str,
                 ids: list[int], names: list[str]) -> dict:
    rpt = classification_report(
        y_true, y_pred, labels=ids, target_names=names,
        output_dict=True, zero_division=0,
    )
    text_rpt = classification_report(
        y_true, y_pred, labels=ids, target_names=names, zero_division=0,
    )
    print(f"\n--- {task} / {model} ---")
    print(text_rpt)
    _save_confusion(y_true, y_pred, task, model, ids, names)
    return {
        "task": task,
        "model": model,
        "accuracy":    rpt["accuracy"],
        "macro_f1":    rpt["macro avg"]["f1-score"],
        "weighted_f1": rpt["weighted avg"]["f1-score"],
    }


def eval_baseline(task: str) -> dict | None:
    path = PATHS["models"] / f"baseline_{task}.joblib"
    if not path.exists():
        return None
    pipe = joblib.load(path)
    test = pd.read_parquet(PATHS["processed"] / f"{task}_test.parquet")
    ids, names = _labels_for(task)
    return _summary_row(test["label"].values, pipe.predict(test["text"]),
                        task, "baseline", ids, names)


def eval_distilbert(task: str, variant: str = "") -> dict | None:
    suffix = f"_{variant}" if variant else ""
    model_dir = PATHS["models"] / f"distilbert_{task}{suffix}"
    if not model_dir.exists():
        return None
    # Lazy imports so the baseline-only path doesn't need torch installed.
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device).eval()
    test = pd.read_parquet(PATHS["processed"] / f"{task}_test.parquet")

    preds: list[int] = []
    bs = 64
    with torch.no_grad():
        for i in range(0, len(test), bs):
            batch = test["text"].iloc[i:i + bs].tolist()
            enc = tok(batch, truncation=True, padding=True,
                      max_length=MAX_LENGTH, return_tensors="pt").to(device)
            logits = mdl(**enc).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
    ids, names = _labels_for(task)
    label = "distilbert" + (f"_{variant}" if variant else "")
    return _summary_row(test["label"].values, preds, task, label, ids, names)


CAVEATS_MD = """
## Validity caveats — read before quoting numbers

1. **`is_dark` is an upper bound — register-contaminated.** dataset.tsv
   negatives are page-chrome HTML fragments ("Pillowcases & Shams", "Write a
   review") while positives are marketing-style copy, so high binary accuracy
   may reflect a register/style detector, not dark-pattern understanding. The
   category model is the more meaningful artifact; treat `is_dark` as a
   coarse gate, not as evidence the model "understands manipulation".

2. **Social Proof F1 is likely optimistic via template memorization.**
   Exact-string dedup does NOT catch templated near-duplicates of the form
   *"Name from City just bought Product about N hours ago"*. The model can
   memorize the template skeleton, inflating Social Proof precision/recall.
   Treated as a known limitation for iter 2 (page-id-level holdout).
"""


def main() -> None:
    rows: list[dict] = []
    for task in ("binary", "multiclass"):
        r = eval_baseline(task)
        if r:
            rows.append(r)
    r = eval_distilbert("binary")
    if r:
        rows.append(r)
    for variant in ("unweighted", "weighted"):
        r = eval_distilbert("multiclass", variant)
        if r:
            rows.append(r)

    if not rows:
        print("No models found. Run src.baseline / src.train_transformer first.")
        return

    df = pd.DataFrame(rows, columns=["task", "model", "accuracy", "macro_f1", "weighted_f1"])
    PATHS["reports"].mkdir(parents=True, exist_ok=True)
    df.to_csv(PATHS["reports"] / "comparison.csv", index=False)

    md = ["# Model comparison (test split)", "",
          df.to_markdown(index=False, floatfmt=".4f"),
          CAVEATS_MD]
    with open(PATHS["reports"] / "comparison.md", "w") as f:
        f.write("\n".join(md))
    print(f"\nWrote {PATHS['reports'] / 'comparison.csv'}")
    print(f"Wrote {PATHS['reports'] / 'comparison.md'}")


if __name__ == "__main__":
    main()
