"""
model.py  —  SleepGuard v2
Random Forest with Leave-One-Patient-Out CV.
Adds Wilcoxon signed-rank test for statistical significance.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score,
                              precision_score, recall_score,
                              roc_curve)
from sklearn.pipeline import Pipeline
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings("ignore")


def lopo_cv(X, y, patient_ids):
    """
    Leave-One-Patient-Out cross-validation.
    Returns per-fold results + per-fold ROC curve data.
    """
    unique_patients = patient_ids.unique()
    results  = []
    roc_data = []

    for test_pid in unique_patients:
        test_mask  = (patient_ids == test_pid).values
        train_mask = ~test_mask

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test  = X[test_mask]
        y_test  = y[test_mask]

        if y_test.sum() == 0 or y_train.sum() == 0:
            continue

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ))
        ])
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auroc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        results.append({
            "patient":   int(test_pid),
            "auroc":     auroc,
            "f1":        f1_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "n_test":    int(len(y_test)),
            "n_hypo":    int(y_test.sum()),
        })
        roc_data.append({"patient": int(test_pid), "fpr": fpr, "tpr": tpr})

    return results, roc_data


def get_feature_importances(X, y, feature_names):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ))
    ])
    clf.fit(X, y)
    imp = clf.named_steps["rf"].feature_importances_
    return pd.Series(imp, index=feature_names).sort_values(ascending=False)


def summarise(results, model_name):
    df = pd.DataFrame(results)
    return {
        "model":        model_name,
        "auroc_mean":   round(df["auroc"].mean(),     3),
        "auroc_std":    round(df["auroc"].std(),      3),
        "auroc_values": df["auroc"].tolist(),
        "f1_mean":      round(df["f1"].mean(),        3),
        "f1_std":       round(df["f1"].std(),         3),
        "recall_mean":  round(df["recall"].mean(),    3),
        "prec_mean":    round(df["precision"].mean(), 3),
    }, df


def run_models(X, y, meta):
    """
    Single model (glucose-only) with LOPO-CV + significance test
    against a majority-class dummy baseline.
    """
    patient_ids = meta["patient_id"].reset_index(drop=True)
    X_vals = X.values
    y_vals = y.values

    print("\n── Random Forest (glucose features) ──")
    results, roc_data = lopo_cv(X_vals, y_vals, patient_ids)
    summary, detail   = summarise(results, "Glucose RF")

    # Dummy baseline: always predict majority class
    # AUROC of 0.5 for each fold
    dummy_aurocs = [0.5] * len(results)
    rf_aurocs    = summary["auroc_values"]

    # Wilcoxon signed-rank test vs dummy (0.5 per fold)
    if len(rf_aurocs) >= 3:
        try:
            differences = [a - 0.5 for a in rf_aurocs]
            stat, pval  = wilcoxon(differences, alternative="greater")
        except Exception:
            stat, pval  = float("nan"), float("nan")
    else:
        stat, pval = float("nan"), float("nan")

    summary["wilcoxon_stat"] = stat
    summary["wilcoxon_pval"] = pval

    # Feature importances
    imp = get_feature_importances(X_vals, y_vals, X.columns.tolist())

    return summary, detail, roc_data, imp
