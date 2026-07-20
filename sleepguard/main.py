"""
main.py  —  SleepGuard v3
Usage:
    python main.py --data "C:\Coding Assignments\OhioT1DM\OhioT1DM\2018"
"""

import os, sys, argparse, math
sys.path.insert(0, os.path.dirname(__file__))

from data_loader         import load_all_patients
from feature_engineering import build_feature_matrix
from model               import run_models
from ablation            import run_ablation
from patient_analysis    import analyze_patient
from visualize           import generate_all


def print_model_table(summaries):
    W = 68
    names = list(summaries.keys())
    print("\n" + "="*W)
    print("  MODEL COMPARISON — Leave-One-Patient-Out CV")
    print("="*W)
    print(f"  {'Model':<22} {'AUROC':>10} {'F1':>8} {'Recall':>8} "
          f"{'Prec':>8} {'p-val':>8}")
    print("-"*W)
    for name, s in summaries.items():
        pval = s["wilcoxon_pval"]
        pstr = f"{pval:.4f}" if not math.isnan(pval) else "N/A"
        sig  = "*" if (not math.isnan(pval) and pval < 0.05) else ""
        print(f"  {name:<22} "
              f"{s['auroc_mean']:.3f}±{s['auroc_std']:.3f}  "
              f"{s['f1_mean']:.3f}    "
              f"{s['recall_mean']:.3f}    "
              f"{s['prec_mean']:.3f}  "
              f"{pstr}{sig}")
    print("="*W)
    print("  * = significant vs chance (p < 0.05, Wilcoxon signed-rank)")


def print_per_patient(summaries, all_details):
    W = 72
    best = max(summaries, key=lambda n: summaries[n]["auroc_mean"])
    detail = all_details[best]
    print(f"\n" + "="*W)
    print(f"  PER-PATIENT RESULTS — {best}")
    print("="*W)
    print(f"  {'Patient':<10} {'AUROC':>8} {'F1':>7} {'Recall':>8} "
          f"{'Thresh':>8} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    print("-"*W)
    for _, row in detail.iterrows():
        flag = "▲" if row["auroc"] > 0.7 else ("~" if row["auroc"] > 0.55 else "▼")
        print(f"  {int(row['patient']):<10} "
              f"{row['auroc']:>8.3f} "
              f"{row['f1']:>7.3f} "
              f"{row['recall']:>8.3f} "
              f"{row['threshold']:>8.2f} "
              f"{int(row['tp']):>4} {int(row['fp']):>4} "
              f"{int(row['fn']):>4} {int(row['tn']):>4}  {flag}")
    print("="*W)


def print_ablation(full_auroc, ablation_df):
    W = 55
    print(f"\n" + "="*W)
    print(f"  FEATURE ABLATION  (full AUROC = {full_auroc:.3f})")
    print("="*W)
    print(f"  {'Feature':<18} {'w/o AUROC':>10} {'Drop':>8}")
    print("-"*W)
    for _, row in ablation_df.iterrows():
        marker = " ◄ key" if row["auroc_drop"] > 0.005 else ""
        print(f"  {row['feature']:<18} {row['auroc_without']:>10.3f} "
              f"{row['auroc_drop']:>+8.4f}{marker}")
    print("="*W)


def main():
    parser = argparse.ArgumentParser(description="SleepGuard v3")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to OhioT1DM 2018 folder")
    parser.add_argument("--skip-ablation", action="store_true",
                        help="Skip ablation study (faster run)")
    args = parser.parse_args()

    print(f"\nData: {args.data}")

    print("\n── Loading patients ──")
    patients = load_all_patients(args.data)
    if not patients:
        print("ERROR: No patients loaded.")
        sys.exit(1)

    print("\n── Extracting features ──")
    X, y, meta = build_feature_matrix(patients)

    print("\n── Training models ──")
    summaries, all_details, all_roc, imp = run_models(X, y, meta)

    print_model_table(summaries)
    print_per_patient(summaries, all_details)

    if not args.skip_ablation:
        full_auroc, ablation_df = run_ablation(X, y, meta)
        print_ablation(full_auroc, ablation_df)
    else:
        full_auroc, ablation_df = 0.0, __import__('pandas').DataFrame()

    analyze_patient(patients, X, y, meta, target_pid=559)

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    for name, detail in all_details.items():
        safe = name.lower().replace(" ", "_")
        detail.to_csv(os.path.join(results_dir, f"results_{safe}.csv"), index=False)
    imp.to_csv(os.path.join(results_dir, "feature_importances.csv"))
    if not ablation_df.empty:
        ablation_df.to_csv(os.path.join(results_dir, "ablation.csv"), index=False)

    generate_all(summaries, all_details, all_roc, imp, full_auroc, ablation_df)

    print(f"\nResults saved to: {os.path.abspath(results_dir)}")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
