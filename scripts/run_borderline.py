#!/usr/bin/env python3
"""PHASE 6 — flag ambiguous near-50% fibroids, with patient-grouped models."""

import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pandas as pd

from figomeas import load_config
from figomeas.borderline import (PREDICTORS, evaluate_grouped, feature_importance,
                                 flag)
from figomeas.config import REPO_ROOT


def main() -> int:
    cfg = load_config()
    src = REPO_ROOT / "results" / "fibroids_uncertainty.parquet"
    if not src.exists():
        src = REPO_ROOT / "results" / "fibroids_figo.parquet"
        print(f"NOTE: {src.name} has no confidence intervals; "
              f"flagging on the point estimate only. Run `make uncertainty` first.\n")
    df = flag(pd.read_parquet(src), cfg)

    out = REPO_ROOT / "results" / "fibroids_borderline.parquet"
    df.to_parquet(out, index=False)
    df.to_csv(out.with_suffix(".csv"), index=False)

    rel = df[df.reliable]
    n_b = int(rel.is_borderline.sum())
    print(f"=== borderline flagging ({len(rel)} reliable fibroids) ===\n")
    print(f"flagged borderline: {n_b} ({100*n_b/len(rel):.1f}%)")
    print(f"  band = +-{cfg['borderline.band_pct']:.0f} points around "
          f"{cfg['figo.intramural_threshold_pct']:.0f}%\n")
    print("why flagged:")
    print(rel.borderline_reason.value_counts().to_string(), "\n")

    at_risk = rel[rel.touch_cavity | rel.touch_serosa]
    n_risk = int(at_risk.is_borderline.sum())
    print(f"among fibroids where the call actually matters (touching a surface, "
          f"so 1-vs-2 or 5-vs-6): {n_risk}/{len(at_risk)} ({100*n_risk/max(len(at_risk),1):.1f}%) "
          f"are ambiguous\n")

    print(f"predictors (none derived from the reference surfaces): {PREDICTORS}\n")
    print("=== patient-grouped cross-validation ===")
    metrics = evaluate_grouped(rel, cfg)
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.3f}"), "\n")

    print("=== permutation importance (random forest, held-out fold) ===")
    imp = feature_importance(rel, cfg)
    print(imp.to_string(index=False, float_format=lambda v: f"{v:.4f}"), "\n")
    metrics.to_csv(REPO_ROOT / "results" / "borderline_metrics.csv", index=False)
    imp.to_csv(REPO_ROOT / "results" / "borderline_importance.csv", index=False)

    best = metrics.iloc[0]
    gates = [
        ("G1  borderline cases exist and are not the whole cohort",
         0.02 <= rel.is_borderline.mean() <= 0.60,
         f"{100*rel.is_borderline.mean():.1f}% flagged"),
        ("G2  grouping is enforced: no patient spans train and test",
         True, "asserted inside every CV split in evaluate_grouped()"),
        ("G3  triage model beats chance under patient-grouped CV",
         float(best.roc_auc) > 0.60,
         f"best {best.model}: ROC AUC {best.roc_auc:.3f}, "
         f"PR AUC {best.pr_auc:.3f} vs base rate {best.base_rate:.3f}"),
        ("G4  no predictor is derived from percent_intramural",
         True, "enforced by assert_no_predictor_leakage()"),
    ]
    print("=== PHASE 6 EXIT GATES ===")
    for name, ok, detail in gates:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}\n         {detail}")
    failed = [n for n, ok, _ in gates if not ok]
    print()
    print("FRAMING: this predicts *which cases need a second look*, not FIGO type.")
    print("The label comes from our own measurement, so the model learns which")
    print("geometries this pipeline cannot resolve. It is triage, not diagnosis.\n")
    if failed:
        print(f"PHASE 6 BLOCKED — {len(failed)} gate(s) failed.")
        return 1
    print("PHASE 6 EXIT GATE: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
