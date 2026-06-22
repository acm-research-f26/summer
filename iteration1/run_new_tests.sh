#!/usr/bin/env bash
# Evaluate the trained models on data/generalization_test.csv — a held-out
# hand-written set used to probe generalization beyond the in-distribution
# test split (where the register-contamination + Social-Proof template caveats
# in reports/comparison.md can inflate scores).
#
# CSV schema:  text,is_dark,category    (category == "none" when is_dark == 0)
#
# Writes:
#   reports/generalization/binary_report.txt
#   reports/generalization/multiclass_report.txt   (dark-only rows)
#   reports/generalization/binary_confusion.csv
#   reports/generalization/multiclass_confusion.csv
#   reports/generalization/predictions.csv         (row-level dump for inspection)

set -euo pipefail

cd "$(dirname "$0")"

CSV="${1:-data/generalization_test.csv}"
if [ ! -f "$CSV" ]; then
  echo "error: $CSV not found" >&2
  exit 1
fi

mkdir -p reports/generalization

CSV_PATH="$CSV" python - <<'PY'
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.inference import predict_batch, EXPECTED_CATEGORY_KEYS

csv_path = Path(os.environ["CSV_PATH"])
out_dir  = Path("reports/generalization")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)
required = {"text", "is_dark", "category"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"CSV missing columns: {sorted(missing)}")

df["text"]     = df["text"].astype(str)
df["is_dark"]  = df["is_dark"].astype(int)
df["category"] = df["category"].astype(str).str.strip()

# Validate category labels (allow "none" only when is_dark == 0).
bad = df[(df["is_dark"] == 1) & (~df["category"].isin(EXPECTED_CATEGORY_KEYS))]
if not bad.empty:
    raise SystemExit(
        f"is_dark=1 rows with unknown category:\n{bad[['text','category']].to_string(index=False)}"
    )

print(f"loaded {len(df)} rows from {csv_path}  "
      f"(dark={int((df['is_dark']==1).sum())}, not_dark={int((df['is_dark']==0).sum())})")

preds = predict_batch(df["text"].tolist())
df["pred_is_dark_prob"] = [p["is_dark"]      for p in preds]
df["pred_is_dark"]      = [int(p["is_dark"] >= 0.5) for p in preds]
df["pred_category"]     = [p["top_category"] for p in preds]

# ---- binary ----
bin_names = ["not_dark", "dark"]
bin_rpt = classification_report(
    df["is_dark"], df["pred_is_dark"],
    labels=[0, 1], target_names=bin_names, digits=4, zero_division=0,
)
bin_cm = pd.DataFrame(
    confusion_matrix(df["is_dark"], df["pred_is_dark"], labels=[0, 1]),
    index=bin_names, columns=bin_names,
)
(out_dir / "binary_report.txt").write_text(bin_rpt)
bin_cm.to_csv(out_dir / "binary_confusion.csv")
print("\n=== binary (is_dark) on full generalization set ===")
print(bin_rpt)
print("confusion:\n", bin_cm, sep="")

# ---- multiclass on dark-only rows (category is undefined for is_dark=0) ----
dark = df[df["is_dark"] == 1]
cat_names = sorted(EXPECTED_CATEGORY_KEYS)
if not dark.empty:
    mc_rpt = classification_report(
        dark["category"], dark["pred_category"],
        labels=cat_names, target_names=cat_names, digits=4, zero_division=0,
    )
    mc_cm = pd.DataFrame(
        confusion_matrix(dark["category"], dark["pred_category"], labels=cat_names),
        index=cat_names, columns=cat_names,
    )
    (out_dir / "multiclass_report.txt").write_text(mc_rpt)
    mc_cm.to_csv(out_dir / "multiclass_confusion.csv")
    print(f"\n=== multiclass (category) on dark-only rows (n={len(dark)}) ===")
    print(mc_rpt)
    print("confusion:\n", mc_cm, sep="")
else:
    print("\n(no dark rows — skipping multiclass evaluation)")

df.to_csv(out_dir / "predictions.csv", index=False)
print(f"\nwrote {out_dir}/  (reports, confusions, predictions.csv)")
PY
