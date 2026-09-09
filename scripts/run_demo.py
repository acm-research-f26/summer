#!/usr/bin/env python3
"""End-to-end demo on synthetic masks. No dataset, no download, no GPU.

Every phantom's FIGO type is known by construction, so this doubles as an
accuracy check: it reports the confusion between derived and true types.
"""

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pandas as pd

from figomeas import load_config
from figomeas.borderline import flag
from figomeas.config import REPO_ROOT
from figomeas.features import extract_patient, partition_residual
from figomeas.figo import derive
from figomeas.io import load_mask
from figomeas.synthetic import build_demo_cohort


def main() -> int:
    cfg = load_config()
    print("building synthetic cohort with known FIGO geometry ...")
    manifest = build_demo_cohort(cfg)

    rows = []
    for m in manifest:
        mask = load_mask(m["mask_path"]).data
        spacing = (m["used_sx"], m["used_sy"], m["used_sz"])
        recs, _, _ = extract_patient(mask, spacing, cfg, m["patient_id"])
        for r in recs:
            r["figo"] = derive(r, cfg)
            r["true_figo"] = m["true_figo"]
            r["true_percent_intramural"] = m["true_percent_intramural"]
            r["correct"] = r["figo"] == m["true_figo"]
            r["abs_error"] = abs(r["percent_intramural"] - m["true_percent_intramural"])
            r["reliable"] = True
            rows.append(r)

    df = pd.DataFrame(rows)
    df["ci_straddles_50"] = False
    df = flag(df, cfg)
    out = REPO_ROOT / cfg["data.demo_root"] / "demo_results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n=== {len(df)} synthetic fibroids across {df.patient_id.nunique()} phantoms ===\n")
    print(df[["patient_id", "true_figo", "figo", "percent_intramural",
              "true_percent_intramural", "abs_error", "correct"]]
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    print(f"\nFIGO type recovered correctly: {int(df.correct.sum())}/{len(df)} "
          f"({100*df.correct.mean():.0f}%)")
    print(f"percent_intramural |error|: median {df.abs_error.median():.1f}, "
          f"max {df.abs_error.max():.1f} points")
    print(f"partition residual max: "
          f"{max(partition_residual(r) for _, r in df.iterrows()):.2e} points")
    print(f"borderline flagged: {int(df.is_borderline.sum())}")

    print("\nconfusion (rows = true, cols = derived):")
    print(pd.crosstab(df.true_figo, df.figo).to_string())

    ok = bool(df.correct.all())
    print(f"\nDEMO {'PASSED' if ok else 'FAILED'}: "
          f"{'every phantom recovered its own FIGO type' if ok else 'see mismatches above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
