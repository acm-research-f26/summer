"""
patient_analysis.py  —  SleepGuard v3
Deep dive into patient 559 (worst performer, AUROC ~0.588)
vs best performer. Compares glucose distributions and pre-sleep
patterns between patients and between hypo/normal nights.
"""

import numpy as np
import pandas as pd


def analyze_patient(patients_data, X, y, meta, target_pid=559):
    """
    Compare target patient vs all others on:
    - Hypoglycemic night rate
    - Mean pre-sleep glucose on hypo vs normal nights
    - Glucose variability
    - Feature distributions
    """
    print(f"\n── Patient {target_pid} Deep Dive ──")

    df = X.copy()
    df["patient_id"] = meta["patient_id"].values
    df["label"]      = y.values

    target = df[df["patient_id"] == target_pid]
    others = df[df["patient_id"] != target_pid]

    print(f"\n  {target_pid} vs others — night counts:")
    print(f"  {'':20} {'P'+str(target_pid):>10} {'Others (mean)':>14}")
    print(f"  {'Total nights':<20} {len(target):>10} {len(others)/5:>14.1f}")
    print(f"  {'Hypo nights':<20} {int(target['label'].sum()):>10} "
          f"{others.groupby('patient_id')['label'].sum().mean():>14.1f}")
    print(f"  {'Hypo rate':<20} {target['label'].mean()*100:>9.1f}% "
          f"{others['label'].mean()*100:>13.1f}%")

    print(f"\n  {target_pid} — feature means on hypo vs normal nights:")
    print(f"  {'Feature':<18} {'Hypo':>10} {'Normal':>10} {'Diff':>8}")
    print(f"  {'-'*48}")

    t_hypo   = target[target["label"] == 1]
    t_normal = target[target["label"] == 0]
    feat_cols = [c for c in X.columns]

    diffs = []
    for feat in feat_cols:
        if feat == "coverage":
            continue
        h_mean = t_hypo[feat].mean()   if len(t_hypo)   > 0 else np.nan
        n_mean = t_normal[feat].mean() if len(t_normal) > 0 else np.nan
        diff   = h_mean - n_mean
        diffs.append((abs(diff), feat, h_mean, n_mean, diff))

    diffs.sort(reverse=True)
    for _, feat, h_mean, n_mean, diff in diffs[:8]:
        print(f"  {feat:<18} {h_mean:>10.2f} {n_mean:>10.2f} {diff:>+8.2f}")

    print(f"\n  Global feature means (hypo vs normal) — all patients:")
    print(f"  {'Feature':<18} {'Hypo':>10} {'Normal':>10} {'Diff':>8}")
    print(f"  {'-'*48}")

    all_hypo   = df[df["label"] == 1]
    all_normal = df[df["label"] == 0]
    for feat in ["g_last", "g_mean", "g_min", "g_slope", "g_accel",
                 "g_modd", "pct_u100", "pct_u80"]:
        h = all_hypo[feat].mean()
        n = all_normal[feat].mean()
        print(f"  {feat:<18} {h:>10.2f} {n:>10.2f} {h-n:>+8.2f}")

    return df
