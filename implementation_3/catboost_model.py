"""
catboost_model.py — CatBoost hyperparameter grid and training routine for MIRA.

Public API
----------
CB_CONFIGS : dict
    Named hyperparameter configurations to grid-search over.

train_catboost(split_data, shared_cv_splits, feature_columns_by_set,
               random_seed) -> tuple
    Run CV grid search for every feature set, refit the winner on the full
    training partition, and return all artefacts needed for evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------
CB_CONFIGS: dict[str, dict] = {
    "shallow_slow": dict(iterations=600, depth=4, learning_rate=0.03, l2_leaf_reg=3.0),
    "balanced":     dict(iterations=300, depth=6, learning_rate=0.05, l2_leaf_reg=3.0),
    "deep_fast":    dict(iterations=200, depth=8, learning_rate=0.10, l2_leaf_reg=1.0),
    "regularized":  dict(iterations=400, depth=5, learning_rate=0.05, l2_leaf_reg=8.0),
}


# ---------------------------------------------------------------------------
# Training routine
# ---------------------------------------------------------------------------
def train_catboost(
    split_data: dict,
    shared_cv_splits: list[tuple[np.ndarray, np.ndarray]],
    feature_columns_by_set: dict[str, list[str]],
    random_seed: int = 42,
) -> tuple[dict, dict, dict, pd.DataFrame]:
    """CV grid-search CB_CONFIGS for every feature set, refit the best.

    Class imbalance is handled internally via ``auto_class_weights="Balanced"``,
    so no manual sample weighting is needed and CV folds are leakage-safe.

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
        Random seed forwarded to CatBoostClassifier.

    Returns
    -------
    trained_models : dict
        ``{(feature_set, "CatBoost"): fitted CatBoostClassifier}``
    selected_configs : dict
        ``{(feature_set, "CatBoost"): config_name}``
    cv_score_distributions : dict
        ``{(feature_set, "CatBoost"): np.ndarray of per-fold macro-F1}``
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

        for config_name, params in CB_CONFIGS.items():
            fold_scores: list[float] = []
            for train_idx, valid_idx in shared_cv_splits:
                model = CatBoostClassifier(
                    **params,
                    auto_class_weights="Balanced",
                    random_seed=random_seed,
                    verbose=0,
                )
                model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                fold_scores.append(
                    f1_score(
                        y_train.iloc[valid_idx],
                        model.predict(X_train.iloc[valid_idx]).ravel(),
                        average="macro",
                    )
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

        final_model = CatBoostClassifier(
            **CB_CONFIGS[best_config],
            auto_class_weights="Balanced",
            random_seed=random_seed,
            verbose=0,
        )
        final_model.fit(X_train, y_train)

        key = (feature_set, "CatBoost")
        trained_models[key] = final_model
        selected_configs[key] = best_config
        cv_score_distributions[key] = config_scores[best_config]

    search_results = pd.DataFrame(search_rows)
    return trained_models, selected_configs, cv_score_distributions, search_results
