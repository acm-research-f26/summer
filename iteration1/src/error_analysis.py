"""Top-3 most-confused class pairs for the multi-class DistilBERT + 10 samples each.

Run:  python -m src.error_analysis

Writes reports/error_analysis.md. Silently skipped if the multi-class checkpoint
is not present yet.
"""
from __future__ import annotations

import random

import pandas as pd
from sklearn.metrics import confusion_matrix

from config import MAX_LENGTH, MULTICLASS_LABELS, PATHS, SEED


def main() -> None:
    model_dir = PATHS["models"] / "distilbert_multiclass"
    if not model_dir.exists():
        print(f"No multi-class model at {model_dir} — skipping.")
        return

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(SEED)

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device).eval()

    test = pd.read_parquet(PATHS["processed"] / "multiclass_test.parquet")
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(test), 64):
            batch = test["text"].iloc[i:i + 64].tolist()
            enc = tok(batch, truncation=True, padding=True,
                      max_length=MAX_LENGTH, return_tensors="pt").to(device)
            logits = mdl(**enc).logits
            p = torch.softmax(logits, dim=-1)
            preds.extend(logits.argmax(-1).cpu().tolist())
            probs.extend(p.cpu().tolist())
    test = test.assign(pred=preds, probs=probs)

    ids   = [i for i, _, _ in MULTICLASS_LABELS]
    names = [d for _, _, d in MULTICLASS_LABELS]
    cm = confusion_matrix(test["label"], test["pred"], labels=ids)

    n = len(ids)
    pair_mass = [((i, j), int(cm[i, j] + cm[j, i]))
                 for i in range(n) for j in range(i + 1, n)]
    pair_mass.sort(key=lambda x: x[1], reverse=True)
    top_pairs = pair_mass[:3]

    lines = ["# Error analysis — multi-class DistilBERT (test split)", "",
             "## Top-3 most-confused class pairs", ""]
    for (i, j), mass in top_pairs:
        lines.append(
            f"- **{names[i]} ↔ {names[j]}** — {mass} off-diagonal "
            f"({cm[i, j]} true {names[i]} → {names[j]}, "
            f"{cm[j, i]} true {names[j]} → {names[i]})"
        )
    lines.append("")
    lines.append("## Sample misclassifications per pair (up to 10 each)")

    for (i, j), _ in top_pairs:
        mis = test[((test["label"] == i) & (test["pred"] == j)) |
                   ((test["label"] == j) & (test["pred"] == i))]
        if len(mis) == 0:
            continue
        sample = mis.sample(min(10, len(mis)), random_state=SEED)
        lines += ["", f"### {names[i]} ↔ {names[j]}", "",
                  "| true | pred | text | probs |",
                  "|---|---|---|---|"]
        for _, r in sample.iterrows():
            probs_str = ", ".join(
                f"{names[k]}={r['probs'][k]:.2f}" for k in range(n)
            )
            text = str(r["text"]).replace("|", "\\|").replace("\n", " ")[:120]
            lines.append(
                f"| {names[r['label']]} | {names[r['pred']]} | {text} | {probs_str} |"
            )

    PATHS["reports"].mkdir(parents=True, exist_ok=True)
    out = PATHS["reports"] / "error_analysis.md"
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
