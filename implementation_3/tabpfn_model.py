"""
tabpfn_model.py — TabPFN training routine for MIRA.

TabPFN is a foundation model (no hyperparameter grid to search).
The CV loop is run to estimate macro-F1 variance on the same shared folds
used by XGBoost and CatBoost, then the final model is refit on the full
training partition.

Public API
----------
train_tabpfn(split_data, shared_cv_splits, feature_columns_by_set,
             random_seed) -> tuple
    Evaluate TabPFN on shared CV folds, refit on full training partition,
    and return all artefacts needed for evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from tabpfn import TabPFNClassifier

try:
    import torch
    _DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
except Exception:
    _DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Training routine
# ---------------------------------------------------------------------------
def train_tabpfn(
    split_data: dict,
    shared_cv_splits: list[tuple[np.ndarray, np.ndarray]],
    feature_columns_by_set: dict[str, list[str]],
    random_seed: int = 42,
) -> tuple[dict, dict, dict, pd.DataFrame]:
    """Evaluate TabPFN on shared CV folds, refit on the full training set.

    Because TabPFN is a foundation model there is no hyperparameter grid to
    search; ``selected_config`` is fixed to ``"n/a (foundation model)"``.

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
        Random seed forwarded to TabPFNClassifier.

    Returns
    -------
    trained_models : dict
        ``{(feature_set, "TabPFN"): fitted TabPFNClassifier}``
    selected_configs : dict
        ``{(feature_set, "TabPFN"): "n/a (foundation model)"}``
    cv_score_distributions : dict
        ``{(feature_set, "TabPFN"): np.ndarray of per-fold macro-F1}``
    search_results : pd.DataFrame
        One-row-per-feature-set summary (feature_set, config, mean, std).
    """
    trained_models: dict = {}
    selected_configs: dict = {}
    cv_score_distributions: dict = {}
    search_rows: list[dict] = []

    for feature_set, data in split_data.items():
        X_train = data["X_train"]
        y_train = data["y_train"]

        # CV fold evaluation (no grid to search — just estimate variance)
        fold_scores: list[float] = []
        for train_idx, valid_idx in shared_cv_splits:
            X_fold_tr = X_train.iloc[train_idx]
            X_fold_va = X_train.iloc[valid_idx]
            y_fold_tr = y_train.iloc[train_idx]
            y_fold_va = y_train.iloc[valid_idx]

            model = TabPFNClassifier(
                random_state=random_seed,
                n_estimators=4,
                auto_scale_n_estimators=False,
                device=_DEVICE,
            )
            model.fit(X_fold_tr, y_fold_tr)
            fold_scores.append(
                f1_score(y_fold_va, model.predict(X_fold_va), average="macro")
            )

        scores_arr = np.asarray(fold_scores)
        search_rows.append({
            "feature_set": feature_set,
            "config": "n/a (foundation model)",
            "cv_macro_f1_mean": scores_arr.mean(),
            "cv_macro_f1_std": scores_arr.std(),
        })

        # Refit on full training partition
        final_model = TabPFNClassifier(
            random_state=random_seed,
            n_estimators=4,
            auto_scale_n_estimators=False,
            device=_DEVICE,
        )
        final_model.fit(X_train, y_train)

        key = (feature_set, "TabPFN")
        trained_models[key] = final_model
        selected_configs[key] = "n/a (foundation model)"
        cv_score_distributions[key] = scores_arr

    search_results = pd.DataFrame(search_rows)
    return trained_models, selected_configs, cv_score_distributions, search_results
