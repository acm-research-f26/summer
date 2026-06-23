"""
main.py  —  SleepGuard pipeline entry point
Usage:
    python main.py                        # runs on synthetic data
    python main.py --data path/to/real/   # runs on real OhioT1DM data
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from data_loader         import load_all_patients
from feature_engineering import build_feature_matrix
from model               import run_models
from visualize           import generate_all


def print_full_results(sum_a, sum_b, detail_a, detail_b, imp_b):
    W = 60

    # ── Overall comparison ────────────────────────────────────────
    print("\n" + "="*W)
    print("  OVERALL RESULTS (Leave-One-Patient-Out CV)")
    print("="*W)
    print(f"  {'Metric':<22} {'Baseline':>12} {'Baseline+HRV':>14}")
    print("-"*W)
    print(f"  {'AUROC (mean±std)':<22} "
          f"{sum_a['auroc_mean']:.3f}±{sum_a['auroc_std']:.3f}   "
          f"  {sum_b['auroc_mean']:.3f}±{sum_b['auroc_std']:.3f}")
    print(f"  {'F1 (mean±std)':<22} "
          f"{sum_a['f1_mean']:.3f}±{sum_a['f1_std']:.3f}   "
          f"  {sum_b['f1_mean']:.3f}±{sum_b['f1_std']:.3f}")
    print(f"  {'Recall (mean)':<22} "
          f"{sum_a['recall_mean']:.3f}         "
          f"  {sum_b['recall_mean']:.3f}")
    print(f"  {'Precision (mean)':<22} "
          f"{sum_a['prec_mean']:.3f}         "
          f"  {sum_b['prec_mean']:.3f}")
    print("="*W)
    delta = sum_b["auroc_mean"] - sum_a["auroc_mean"]
    print(f"\n  HRV contribution to AUROC: {delta:+.3f}")
    if delta > 0:
        print("  >> HRV features IMPROVED prediction over baseline.")
    elif delta < 0:
        print("  >> HRV features HURT prediction vs baseline.")
    else:
        print("  >> HRV features had NO effect.")

    # ── Per-patient breakdown ─────────────────────────────────────
    print("\n" + "="*W)
    print("  PER-PATIENT AUROC")
    print("="*W)
    print(f"  {'Patient':<12} {'Baseline':>10} {'Base+HRV':>10} {'Delta':>8} {'Hypo nights':>12}")
    print("-"*W)

    import pandas as pd
    merged = detail_a[["patient","auroc","n_hypo"]].merge(
        detail_b[["patient","auroc"]], on="patient", suffixes=("_base","_hrv")
    )
    for _, row in merged.iterrows():
        d = row["auroc_hrv"] - row["auroc_base"]
        flag = "▲" if d > 0.01 else ("▼" if d < -0.01 else "~")
        print(f"  {int(row['patient']):<12} "
              f"{row['auroc_base']:>10.3f} "
              f"{row['auroc_hrv']:>10.3f} "
              f"{d:>+8.3f} {flag}"
              f"  {int(row['n_hypo']):>8} nights")

    # ── Feature importances ───────────────────────────────────────
    print("\n" + "="*W)
    print("  TOP 10 FEATURE IMPORTANCES (Model B — Baseline+HRV)")
    print("="*W)
    for i, (feat, val) in enumerate(imp_b.head(10).items(), 1):
        tag = " [HRV]" if feat.startswith("hrv") or feat.startswith("hr_") else ""
        bar = "█" * int(val * 200)
        print(f"  {i:>2}. {feat:<20}{tag:<7} {val:.4f}  {bar}")

    print("\n" + "="*W)


def main():
    parser = argparse.ArgumentParser(description="SleepGuard pipeline")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to data directory. Defaults to synthetic data.")
    args = parser.parse_args()

    if args.data:
        data_dir = args.data
        print(f"Using real data from: {data_dir}")
    else:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
        print("No --data path provided. Using synthetic data.")
        train_dir = os.path.join(data_dir, "train")
        if not os.path.isdir(train_dir) or len(os.listdir(train_dir)) == 0:
            print("Generating synthetic data...")
            from generate_synthetic_data import main as gen
            gen()

    print("\n── Loading patient data ──")
    patients = load_all_patients(data_dir)

    print("\n── Extracting features ──")
    X_baseline, X_hrv, y, meta = build_feature_matrix(patients)

    print("\n── Training & evaluating models ──")
    sum_a, sum_b, detail_a, detail_b, imp_b = run_models(X_baseline, X_hrv, y, meta)

    print_full_results(sum_a, sum_b, detail_a, detail_b, imp_b)

    # Save PNGs — always overwrites same files, no duplicates
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    detail_a.to_csv(os.path.join(results_dir, "per_patient_baseline.csv"), index=False)
    detail_b.to_csv(os.path.join(results_dir, "per_patient_hrv.csv"),      index=False)
    imp_b.to_csv(   os.path.join(results_dir, "feature_importances.csv"))
    generate_all(sum_a, sum_b, detail_a, detail_b, imp_b)
    print(f"\nPNGs saved/overwritten in: {os.path.abspath(results_dir)}")


if __name__ == "__main__":
    main()