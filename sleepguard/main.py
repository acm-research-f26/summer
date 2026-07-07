"""
main.py  —  SleepGuard v2
Usage:
    python main.py --data "C:\Coding Assignments\OhioT1DM\OhioT1DM\2018"
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from data_loader        import load_all_patients
from feature_engineering import build_feature_matrix
from model              import run_models
from visualize          import generate_all


def print_results(summary, detail):
    W = 58
    print("\n" + "="*W)
    print("  SLEEPGUARD v2 — REAL OhioT1DM 2018 RESULTS")
    print("  Glucose-Only Pre-Sleep Feature Model")
    print("="*W)
    print(f"  {'AUROC (mean±std)':<28} "
          f"{summary['auroc_mean']:.3f} ± {summary['auroc_std']:.3f}")
    print(f"  {'F1 (mean±std)':<28} "
          f"{summary['f1_mean']:.3f} ± {summary['f1_std']:.3f}")
    print(f"  {'Recall (mean)':<28} {summary['recall_mean']:.3f}")
    print(f"  {'Precision (mean)':<28} {summary['prec_mean']:.3f}")
    print("-"*W)

    pval = summary["wilcoxon_pval"]
    stat = summary["wilcoxon_stat"]
    sig  = "YES (p < 0.05)" if (not __import__('math').isnan(pval) and pval < 0.05) else \
           "NO"             if not __import__('math').isnan(pval) else "N/A (too few folds)"
    print(f"  Wilcoxon vs chance (0.5):")
    print(f"    statistic = {stat:.3f}   p-value = {pval:.4f}")
    print(f"    Significant? {sig}")
    print("="*W)

    print("\n" + "="*W)
    print("  PER-PATIENT BREAKDOWN")
    print("="*W)
    print(f"  {'Patient':<10} {'AUROC':>8} {'F1':>7} {'Recall':>8} {'Hypo nights':>12}")
    print("-"*W)
    for _, row in detail.iterrows():
        beat = "▲" if row["auroc"] > 0.6 else ("~" if row["auroc"] > 0.5 else "▼")
        print(f"  {int(row['patient']):<10} "
              f"{row['auroc']:>8.3f} "
              f"{row['f1']:>7.3f} "
              f"{row['recall']:>8.3f} "
              f"{int(row['n_hypo']):>8} nights  {beat}")
    print("="*W)


def print_importances(imp):
    W = 58
    print("\n" + "="*W)
    print("  TOP FEATURE IMPORTANCES")
    print("="*W)
    for i, (feat, val) in enumerate(imp.items(), 1):
        bar = "█" * int(val * 300)
        print(f"  {i:>2}. {feat:<18} {val:.4f}  {bar}")
    print("="*W)


def main():
    parser = argparse.ArgumentParser(description="SleepGuard v2")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to OhioT1DM 2018 folder (contains train/ subdir)")
    args = parser.parse_args()

    print(f"\nData path: {args.data}")

    print("\n── Loading patients ──")
    patients = load_all_patients(args.data)
    if not patients:
        print("ERROR: No patient data loaded. Check your --data path.")
        sys.exit(1)

    print("\n── Extracting features ──")
    X, y, meta = build_feature_matrix(patients)
    if len(X) == 0:
        print("ERROR: No nights passed quality filter. Check data.")
        sys.exit(1)

    print("\n── Training model ──")
    summary, detail, roc_data, imp = run_models(X, y, meta)

    print_results(summary, detail)
    print_importances(imp)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    detail.to_csv(os.path.join(results_dir, "per_patient_results.csv"), index=False)
    imp.to_csv(   os.path.join(results_dir, "feature_importances.csv"))

    generate_all(summary, detail, roc_data, imp)

    print(f"\nResults saved to: {os.path.abspath(results_dir)}")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
