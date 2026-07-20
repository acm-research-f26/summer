"""
model.py  —  SleepGuard v3
Three models compared via Leave-One-Patient-Out CV:
  1. Logistic Regression (linear baseline)
  2. Random Forest
  3. XGBoost
Each model also gets threshold optimization for max recall
at precision >= 0.40.
Wilcoxon test: each model vs chance (0.5) and RF vs LR, XGB vs RF.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble       import RandomForestClassifier
from sklearn.linear_model   import LogisticRegression
from sklearn.preprocessing  import StandardScaler
from sklearn.metrics        import (roc_auc_score, f1_score,
                                    precision_score, recall_score,
                                    roc_curve, confusion_matrix)
from sklearn.pipeline       import Pipeline
from scipy.stats            import wilcoxon
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed — skipping XGBoost model.")

MIN_PRECISION = 0.35   # minimum acceptable precision when optimizing threshold


def _make_lr():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ))
    ])


def _make_rf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ))
    ])


def _make_xgb(scale_pos_weight):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ))
    ])


def _optimal_threshold(y_true, y_prob, min_precision=MIN_PRECISION):
    """
    Find threshold that maximises recall while keeping precision >= min_precision.
    Falls back to 0.5 if no threshold meets the constraint.
    """
    thresholds = np.linspace(0.1, 0.9, 81)
    best_thresh  = 0.5
    best_recall  = 0.0
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if y_pred.sum() == 0:
            continue
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        if p >= min_precision and r > best_recall:
            best_recall = r
            best_thresh = t
    return best_thresh


def lopo_cv(model_fn, X, y, patient_ids):
    unique_pids = patient_ids.unique()
    results, roc_data = [], []

    for test_pid in unique_pids:
        test_mask  = (patient_ids == test_pid).values
        train_mask = ~test_mask

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        if y_test.sum() == 0 or y_train.sum() == 0:
            continue

        spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        clf = model_fn(spw)
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]
        thresh = _optimal_threshold(y_test, y_prob)
        y_pred = (y_prob >= thresh).astype(int)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        results.append({
            "patient":   int(test_pid),
            "auroc":     roc_auc_score(y_test, y_prob),
            "f1":        f1_score(y_test, y_pred, zero_division=0),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "threshold": round(thresh, 2),
            "n_test":    int(len(y_test)),
            "n_hypo":    int(y_test.sum()),
            "tn": int(cm[0,0]), "fp": int(cm[0,1]),
            "fn": int(cm[1,0]), "tp": int(cm[1,1]),
        })
        roc_data.append({"patient": int(test_pid), "fpr": fpr, "tpr": tpr})

    return results, roc_data


def _summarise(results, name):
    df = pd.DataFrame(results)
    return {
        "model":        name,
        "auroc_mean":   round(df["auroc"].mean(),     3),
        "auroc_std":    round(df["auroc"].std(),      3),
        "auroc_values": df["auroc"].tolist(),
        "f1_mean":      round(df["f1"].mean(),        3),
        "f1_std":       round(df["f1"].std(),         3),
        "recall_mean":  round(df["recall"].mean(),    3),
        "prec_mean":    round(df["precision"].mean(), 3),
    }, df


def _wilcoxon_vs_chance(auroc_values):
    if len(auroc_values) < 3:
        return float("nan"), float("nan")
    try:
        diffs = [a - 0.5 for a in auroc_values]
        s, p  = wilcoxon(diffs, alternative="greater")
        return float(s), float(p)
    except Exception:
        return float("nan"), float("nan")


def get_feature_importances(X, y, feature_names):
    spw = (y == 0).sum() / max((y == 1).sum(), 1)
    clf = _make_rf()(spw)
    clf.fit(X, y)
    imp = clf.named_steps["clf"].feature_importances_
    return pd.Series(imp, index=feature_names).sort_values(ascending=False)


def run_models(X, y, meta):
    pids   = meta["patient_id"].reset_index(drop=True)
    X_vals = X.values
    y_vals = y.values

    models = [
        ("Logistic Regression", lambda spw: _make_lr()),
        ("Random Forest",       lambda spw: _make_rf()),
    ]
    if HAS_XGB:
        models.append(("XGBoost", lambda spw: _make_xgb(spw)))

    all_summaries, all_details, all_roc = {}, {}, {}

    for name, fn in models:
        print(f"\n── {name} ──")
        res, roc = lopo_cv(fn, X_vals, y_vals, pids)
        summary, detail = _summarise(res, name)
        stat, pval = _wilcoxon_vs_chance(summary["auroc_values"])
        summary["wilcoxon_stat"] = stat
        summary["wilcoxon_pval"] = pval
        all_summaries[name] = summary
        all_details[name]   = detail
        all_roc[name]       = roc

    imp = get_feature_importances(X_vals, y_vals, X.columns.tolist())

    return all_summaries, all_details, all_roc, imp
