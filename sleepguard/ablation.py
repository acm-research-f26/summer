"""
ablation.py  —  SleepGuard v3
Feature ablation study: remove one feature at a time and measure
AUROC drop vs the full model. Shows which features are actually
carrying the model vs which are noise.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import roc_auc_score
from sklearn.pipeline      import Pipeline
import warnings
warnings.filterwarnings("ignore")


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


def _lopo_auroc(X, y, patient_ids):
    unique_pids = patient_ids.unique()
    aurocs = []
    for test_pid in unique_pids:
        test_mask  = (patient_ids == test_pid).values
        train_mask = ~test_mask
        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]
        if y_test.sum() == 0 or y_train.sum() == 0:
            continue
        clf = _make_rf()
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        aurocs.append(roc_auc_score(y_test, y_prob))
    return float(np.mean(aurocs)) if aurocs else 0.0


def run_ablation(X, y, meta):
    """
    For each feature, compute AUROC with that feature removed.
    Returns DataFrame sorted by impact (largest drop first).
    """
    pids        = meta["patient_id"].reset_index(drop=True)
    X_vals      = X.values
    y_vals      = y.values
    feature_names = X.columns.tolist()

    print("\n── Feature Ablation Study ──")
    full_auroc = _lopo_auroc(X_vals, y_vals, pids)
    print(f"  Full model AUROC: {full_auroc:.3f}")

    rows = []
    for i, feat in enumerate(feature_names):
        cols_kept = [j for j in range(len(feature_names)) if j != i]
        X_drop    = X_vals[:, cols_kept]
        auroc     = _lopo_auroc(X_drop, y_vals, pids)
        drop      = full_auroc - auroc
        rows.append({
            "feature":    feat,
            "auroc_without": round(auroc, 3),
            "auroc_drop":    round(drop,  4),
        })
        print(f"  drop {feat:<18} → AUROC {auroc:.3f}  (Δ {drop:+.4f})")

    df = pd.DataFrame(rows).sort_values("auroc_drop", ascending=False)
    return full_auroc, df
