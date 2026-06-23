"""
model.py
Trains and evaluates two Random Forest classifiers:
  Model A — baseline features only  (glucose + insulin + carbs)
  Model B — baseline + HRV features

Evaluation: AUROC, F1, precision, recall via leave-one-patient-out CV.
Handles class imbalance with class_weight='balanced'.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score,
                              precision_score, recall_score,
                              confusion_matrix)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


# ── Leave-One-Patient-Out CV ───────────────────────────────────────────────────

def lopo_cv(X, y, patient_ids):
    """
    Leave-One-Patient-Out cross validation.
    For each patient, train on all others, test on that patient.
    Returns list of per-fold result dicts.
    """
    unique_patients = patient_ids.unique()
    results = []

    for test_pid in unique_patients:
        test_mask  = patient_ids == test_pid
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        if y_test.sum() == 0:        # no positives in test fold — skip
            continue

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("rf",     RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ))
        ])

        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        results.append({
            "patient":   test_pid,
            "auroc":     roc_auc_score(y_test, y_prob),
            "f1":        f1_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "n_test":    int(len(y_test)),
            "n_hypo":    int(y_test.sum()),
        })

    return results


# ── Feature importance (train on full dataset) ─────────────────────────────────

def get_feature_importances(X, y, feature_names):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf",     RandomForestClassifier(
            n_estimators=200,
            max_depth=6,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ))
    ])
    clf.fit(X, y)
    importances = clf.named_steps["rf"].feature_importances_
    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


# ── Summarise results ──────────────────────────────────────────────────────────

def summarise(results, model_name):
    df = pd.DataFrame(results)
    summary = {
        "model":     model_name,
        "auroc_mean": round(df["auroc"].mean(), 3),
        "auroc_std":  round(df["auroc"].std(),  3),
        "f1_mean":    round(df["f1"].mean(),    3),
        "f1_std":     round(df["f1"].std(),     3),
        "recall_mean":round(df["recall"].mean(),3),
        "prec_mean":  round(df["precision"].mean(),3),
    }
    return summary, df


# ── Main entry point ───────────────────────────────────────────────────────────

def run_models(X_baseline, X_hrv, y, meta):
    patient_ids = meta["patient_id"]

    print("\n── Model A: Baseline features only ──")
    res_a = lopo_cv(X_baseline.values, y.values, patient_ids)
    sum_a, detail_a = summarise(res_a, "Baseline")

    print("\n── Model B: Baseline + HRV features ──")
    res_b = lopo_cv(X_hrv.values, y.values, patient_ids)
    sum_b, detail_b = summarise(res_b, "Baseline + HRV")

    # Feature importances on full set
    imp_b = get_feature_importances(X_hrv.values, y.values, X_hrv.columns.tolist())

    return sum_a, sum_b, detail_a, detail_b, imp_b


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loader import load_all_patients
    from feature_engineering import build_feature_matrix
    synthetic_dir = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")
    patients = load_all_patients(synthetic_dir)
    Xb, Xh, y, meta = build_feature_matrix(patients)
    run_models(Xb, Xh, y, meta)
