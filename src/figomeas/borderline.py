"""Borderline (ambiguous near-50%) flagging + patient-grouped models (Phase 6).

The deliverable: which fibroids sit close enough to the 50% line that the FIGO
1-vs-2 or 5-vs-6 call is not actually determined by the imaging. This is the
radiologists' stated pain point, and it is the one place a model belongs in this
project.

What the model is, and what it is not
-------------------------------------
The label ``is_borderline`` is derived from **our own measurement** -- either the
point estimate falls within a band of 50%, or the Phase 5 confidence interval
crosses it. So a model trained to predict it is not learning clinical truth; it
is learning *which fibroid geometries this pipeline cannot resolve*. That is a
genuinely useful triage signal ("this one needs a human to look again") and it is
honest, but it must never be reported as FIGO classification accuracy.

To keep it a triage tool rather than a restatement of the answer, the predictors
are restricted to features obtainable **without** the reference-surface
reconstruction: size, shape, and how well the fibroid is sampled through-plane.
Feeding percent_intramural back in would be circular -- the label is a function
of it -- so those columns are excluded by construction, not by convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Deliberately excludes percent_intramural, frac_*, dist_*, and anything else
# derived from the Phase 2 reference surfaces.
PREDICTORS = [
    "volume_mm3", "max_diameter_mm", "n_slices_spanned",
    "elongation", "flatness", "solidity", "sphericity",
    "sz", "anisotropy",
]

LEAKY = ("percent_intramural", "frac_intracavitary", "frac_extraserosal",
         "dist_to_cavity_mm", "dist_to_serosa_mm", "ci_", "p_median",
         "touch_", "figo", "near_50", "submucosal", "subserosal", "reliable")


def flag(table: pd.DataFrame, cfg) -> pd.DataFrame:
    """Add ``is_borderline`` and the reason it was flagged."""
    df = table.copy()
    thr = float(cfg["figo.intramural_threshold_pct"])
    band = float(cfg["borderline.band_pct"])

    df["dist_from_threshold"] = (df.percent_intramural - thr).abs()
    near = df.dist_from_threshold <= band
    straddle = (df.ci_straddles_50.fillna(False).astype(bool)
                if "ci_straddles_50" in df else pd.Series(False, index=df.index))
    if not bool(cfg["borderline.use_ci_straddle"]):
        straddle = pd.Series(False, index=df.index)

    df["near_threshold"] = near
    df["ci_crosses_threshold"] = straddle
    df["is_borderline"] = near | straddle
    df["borderline_reason"] = np.select(
        [near & straddle, near & ~straddle, ~near & straddle],
        ["point estimate and interval", "point estimate", "interval"],
        default="not borderline")
    return df


def assert_no_predictor_leakage(features: list[str]) -> None:
    """Fail loudly if a column derived from the measurement reaches the model."""
    bad = [f for f in features if any(f.startswith(p) or p in f for p in LEAKY)]
    if bad:
        raise ValueError(f"predictors leak the label: {bad}")


def build_design_matrix(df: pd.DataFrame):
    """Predictors + label + grouping vector, with unusable rows dropped."""
    d = df.copy()
    d["anisotropy"] = d.sz / d.sx
    feats = [f for f in PREDICTORS if f in d.columns]
    assert_no_predictor_leakage(feats)

    keep = d[feats].notna().all(axis=1) & d.is_borderline.notna()
    d = d[keep]
    X = d[feats].to_numpy(dtype=float)
    y = d.is_borderline.to_numpy(dtype=int)
    groups = d.patient_id.to_numpy()
    return X, y, groups, feats, d


def _models(cfg, seed):
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    catalogue = {
        "logistic": Pipeline([("scale", StandardScaler()),
                              ("clf", LogisticRegression(max_iter=2000,
                                                         class_weight="balanced",
                                                         random_state=seed))]),
        "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                                class_weight="balanced_subsample",
                                                random_state=seed, n_jobs=1),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }
    return {k: v for k, v in catalogue.items() if k in cfg["borderline.models"]}


def evaluate_grouped(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Patient-grouped cross-validation.

    No patient's fibroids may span train and test: fibroids from one uterus share
    a reconstruction, a slice thickness and an annotator, so splitting them would
    inflate every metric here.
    """
    from sklearn.base import clone
    from sklearn.metrics import (average_precision_score, brier_score_loss,
                                 roc_auc_score)
    from sklearn.model_selection import StratifiedGroupKFold

    X, y, groups, feats, _ = build_design_matrix(df)
    seed = int(cfg["seed"])
    n_splits = int(cfg["borderline.cv_folds"])
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    rows = []
    for name, model in _models(cfg, seed).items():
        oof = np.full(len(y), np.nan)
        for tr, te in cv.split(X, y, groups):
            assert not (set(groups[tr]) & set(groups[te])), "patient leaked across folds"
            m = clone(model)          # a fresh estimator per fold, never a refit one
            m.fit(X[tr], y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
        rows.append({
            "model": name,
            "roc_auc": float(roc_auc_score(y, oof)),
            "pr_auc": float(average_precision_score(y, oof)),
            "brier": float(brier_score_loss(y, oof)),
            "base_rate": float(y.mean()),
            "n": int(len(y)),
        })
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def feature_importance(df: pd.DataFrame, cfg) -> pd.DataFrame:
    """Permutation importance of each predictor, grouped-CV honest."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import StratifiedGroupKFold

    X, y, groups, feats, _ = build_design_matrix(df)
    seed = int(cfg["seed"])
    cv = StratifiedGroupKFold(n_splits=int(cfg["borderline.cv_folds"]),
                              shuffle=True, random_state=seed)
    tr, te = next(cv.split(X, y, groups))
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                class_weight="balanced_subsample",
                                random_state=seed, n_jobs=1).fit(X[tr], y[tr])
    imp = permutation_importance(rf, X[te], y[te], n_repeats=20,
                                 random_state=seed, scoring="roc_auc")
    return (pd.DataFrame({"feature": feats, "importance": imp.importances_mean,
                          "sd": imp.importances_std})
            .sort_values("importance", ascending=False))
