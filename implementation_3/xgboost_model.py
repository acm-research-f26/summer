"""
xgboost_model.py — XGBoost hyperparameter grid and training routine for MIRA.

Public API
----------
XGB_CONFIGS : dict
    Named hyperparameter configurations to grid-search over.

train_xgboost(split_data, shared_cv_splits, feature_columns_by_set,
              random_seed) -> tuple
    Run CV grid search for every feature set, refit the winner on the full
    training partition, and return all artefacts needed for evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------
XGB_CONFIGS: dict[str, dict] = {
    "shallow_slow": dict(
        n_estimators=600, max_depth=3, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    ),
    "balanced": dict(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8, min_child_weight=1,
    ),
    "deep_fast": dict(
        n_estimators=200, max_depth=8, learning_rate=0.10,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=1,
    ),
    "regularized": dict(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        reg_alpha=0.5, reg_lambda=2.0,
    ),
}


# ---------------------------------------------------------------------------
# Training routine
# ---------------------------------------------------------------------------
def train_xgboost(
    split_data: dict,
    shared_cv_splits: list[tuple[np.ndarray, np.ndarray]],
    feature_columns_by_set: dict[str, list[str]],
    random_seed: int = 42,
) -> tuple[dict, dict, dict, pd.DataFrame]:
    """CV grid-search XGB_CONFIGS for every feature set, refit the best.

    Parameters
    ----------
    split_data : dict
        Mapping of feature-set name → ``{"X_train", "X_test", "y_train",
        "y_test"}`` DataFrames/Series with pre-aligned indices.
    shared_cv_splits : list of (train_idx, valid_idx) arrays
        A single, shared stratified-K-fold assignment used for all feature
        sets so CV results are directly comparable.
    feature_columns_by_set : dict
        Mapping of feature-set name → list of predictor column names.
    random_seed : int
        Random seed forwarded to XGBClassifier.

    Returns
    -------
    trained_models : dict
        ``{(feature_set, "XGBoost"): fitted XGBClassifier}``
    selected_configs : dict
        ``{(feature_set, "XGBoost"): config_name}``
    cv_score_distributions : dict
        ``{(feature_set, "XGBoost"): np.ndarray of per-fold macro-F1}``
    search_results : pd.DataFrame
        Full grid-search summary (feature_set, config, mean, std).
    """
    trained_models: dict = {}
    selected_configs: dict = {}
    cv_score_distributions: dict = {}
    search_rows: list[dict] = []

    for feature_set, data in split_data.items():
        X_train = data["X_train"]
        y_train = data["y_train"]
        config_scores: dict[str, np.ndarray] = {}

        for config_name, params in XGB_CONFIGS.items():
            fold_scores: list[float] = []
            for train_idx, valid_idx in shared_cv_splits:
                X_fold_tr = X_train.iloc[train_idx]
                X_fold_va = X_train.iloc[valid_idx]
                y_fold_tr = y_train.iloc[train_idx]
                y_fold_va = y_train.iloc[valid_idx]

                model = XGBClassifier(
                    **params,
                    eval_metric="mlogloss",
                    random_state=random_seed,
                    verbosity=0,
                )
                model.fit(
                    X_fold_tr,
                    y_fold_tr,
                    sample_weight=compute_sample_weight("balanced", y_fold_tr),
                )
                fold_scores.append(
                    f1_score(y_fold_va, model.predict(X_fold_va), average="macro")
                )

            scores_arr = np.asarray(fold_scores)
            config_scores[config_name] = scores_arr
            search_rows.append({
                "feature_set": feature_set,
                "config": config_name,
                "cv_macro_f1_mean": scores_arr.mean(),
                "cv_macro_f1_std": scores_arr.std(),
            })

        best_config = max(config_scores, key=lambda n: config_scores[n].mean())

        final_model = XGBClassifier(
            **XGB_CONFIGS[best_config],
            eval_metric="mlogloss",
            random_state=random_seed,
            verbosity=0,
        )
        final_model.fit(
            X_train,
            y_train,
            sample_weight=compute_sample_weight("balanced", y_train),
        )

        key = (feature_set, "XGBoost")
        trained_models[key] = final_model
        selected_configs[key] = best_config
        cv_score_distributions[key] = config_scores[best_config]

    search_results = pd.DataFrame(search_rows)
    return trained_models, selected_configs, cv_score_distributions, search_results
