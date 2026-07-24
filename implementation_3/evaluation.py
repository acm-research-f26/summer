"""
evaluation.py — metrics comparison, confusion matrices, and feature importance
for MIRA implementation_3.

Public API
----------
compute_comparison_results(...)
    Build the primary comparison table: CV scores, held-out macro/weighted F1,
    accuracy, per-class F1, and a majority-class baseline row.

compute_feature_importance(...)
    Collect native (built-in) and permutation importances for every
    feature-set / model combination and return a tidy DataFrame plus
    a per-group ranking column.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from config import TIER_LABELS


def compute_comparison_results(
    split_data: dict,
    trained_models: dict,
    selected_configs: dict,
    cv_score_distributions: dict,
    feature_columns_by_set: dict[str, list[str]],
    y_train: pd.Series,
    y_test: pd.Series,
    shared_cv_splits: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, dict, dict]:
    """Build the comparison table and compute confusion matrices.

    Parameters
    ----------
    split_data : dict
        ``{feature_set: {"X_train", "X_test", "y_train", "y_test"}}``
    trained_models : dict
        ``{(feature_set, model_name): fitted model}``
    selected_configs : dict
        ``{(feature_set, model_name): config_name}``
    cv_score_distributions : dict
        ``{(feature_set, model_name): np.ndarray of per-fold macro-F1}``
    feature_columns_by_set : dict
        ``{feature_set: [column names]}``
    y_train, y_test : pd.Series
        Shared target arrays (integer class labels 0 / 1 / 2).
    shared_cv_splits : list of (train_idx, valid_idx)
        CV fold indices used to compute baseline CV scores.

    Returns
    -------
    comparison_results : pd.DataFrame
        One row per (feature_set, model) pair, sorted by test_macro_f1.
    test_predictions : dict
        ``{(feature_set, model_name): np.ndarray of integer predictions}``
    confusion_matrices : dict
        ``{(feature_set, model_name): np.ndarray confusion matrix}``
    """
    tier_labels_sorted = sorted(TIER_LABELS)
    tier_names = [TIER_LABELS[k] for k in tier_labels_sorted]

    result_rows: list[dict] = []
    test_predictions: dict = {}
    confusion_matrices_out: dict = {}

    # Majority-class baseline
    baseline_cv_scores = []
    for train_idx, valid_idx in shared_cv_splits:
        fold_majority = y_train.iloc[train_idx].mode().iloc[0]
        fold_pred = np.full(len(valid_idx), fold_majority)
        baseline_cv_scores.append(
            f1_score(y_train.iloc[valid_idx], fold_pred, average="macro")
        )
    baseline_cv_scores = np.asarray(baseline_cv_scores)
    majority_class = y_train.mode().iloc[0]
    baseline_pred = np.full(len(y_test), majority_class)
    baseline_report = classification_report(
        y_test,
        baseline_pred,
        labels=tier_labels_sorted,
        target_names=tier_names,
        output_dict=True,
        zero_division=0,
    )
    result_rows.append({
        "feature_set": "Shared rows",
        "model": "Majority baseline",
        "selected_config": f"predict {TIER_LABELS[majority_class]}",
        "cv_macro_f1_mean": baseline_cv_scores.mean(),
        "cv_macro_f1_std": baseline_cv_scores.std(),
        "test_macro_f1": f1_score(y_test, baseline_pred, average="macro"),
        "test_weighted_f1": f1_score(y_test, baseline_pred, average="weighted"),
        "test_accuracy": accuracy_score(y_test, baseline_pred),
        **{
            f"f1_{name.lower().replace('/', '_')}": baseline_report[name]["f1-score"]
            for name in tier_names
        },
    })

    # Model rows
    for feature_set in feature_columns_by_set:
        data = split_data[feature_set]
        for model_name in ["XGBoost", "CatBoost"]:
            key = (feature_set, model_name)
            pred = np.asarray(
                trained_models[key].predict(data["X_test"])
            ).ravel().astype(int)
            report = classification_report(
                data["y_test"],
                pred,
                labels=tier_labels_sorted,
                target_names=tier_names,
                output_dict=True,
                zero_division=0,
            )
            scores = cv_score_distributions[key]
            result_rows.append({
                "feature_set": feature_set,
                "model": model_name,
                "selected_config": selected_configs[key],
                "cv_macro_f1_mean": scores.mean(),
                "cv_macro_f1_std": scores.std(),
                "test_macro_f1": f1_score(data["y_test"], pred, average="macro"),
                "test_weighted_f1": f1_score(data["y_test"], pred, average="weighted"),
                "test_accuracy": accuracy_score(data["y_test"], pred),
                **{
                    f"f1_{name.lower().replace('/', '_')}": report[name]["f1-score"]
                    for name in tier_names
                },
            })
            test_predictions[key] = pred
            confusion_matrices_out[key] = confusion_matrix(
                data["y_test"], pred, labels=tier_labels_sorted
            )

    comparison_results = (
        pd.DataFrame(result_rows)
        .sort_values("test_macro_f1", ascending=False)
        .reset_index(drop=True)
    )
    return comparison_results, test_predictions, confusion_matrices_out


def compute_feature_importance(
    split_data: dict,
    trained_models: dict,
    feature_columns_by_set: dict[str, list[str]],
    random_seed: int = 42,
    n_repeats: int = 30,
) -> pd.DataFrame:
    """Collect native and permutation importance for every model.

    Parameters
    ----------
    split_data : dict
        ``{feature_set: {"X_test", "y_test", ...}}``
    trained_models : dict
        ``{(feature_set, model_name): fitted model}``
    feature_columns_by_set : dict
        ``{feature_set: [column names]}``
    random_seed : int
        Forwarded to ``permutation_importance``.
    n_repeats : int
        Number of permutation repeats (default 30).

    Returns
    -------
    pd.DataFrame
        Tidy frame with columns: feature_set, model, method, feature,
        importance, importance_std, rank_within_method.
    """
    importance_rows: list[dict] = []

    for feature_set, feature_columns in feature_columns_by_set.items():
        data = split_data[feature_set]
        for model_name in ["XGBoost", "CatBoost"]:
            key = (feature_set, model_name)
            model = trained_models[key]

            # Native importance
            native_values = (
                model.feature_importances_
                if model_name == "XGBoost"
                else model.get_feature_importance()
            )

            # Permutation importance on held-out test set
            perm = permutation_importance(
                model,
                data["X_test"],
                data["y_test"],
                scoring="f1_macro",
                n_repeats=n_repeats,
                random_state=random_seed,
                n_jobs=-1,
            )

            for feature, native, perm_mean, perm_std in zip(
                feature_columns,
                native_values,
                perm.importances_mean,
                perm.importances_std,
            ):
                importance_rows.extend([
                    {
                        "feature_set": feature_set,
                        "model": model_name,
                        "method": "Native",
                        "feature": feature,
                        "importance": float(native),
                        "importance_std": float("nan"),
                    },
                    {
                        "feature_set": feature_set,
                        "model": model_name,
                        "method": "Permutation (test macro-F1)",
                        "feature": feature,
                        "importance": float(perm_mean),
                        "importance_std": float(perm_std),
                    },
                ])

    df = pd.DataFrame(importance_rows)
    df["rank_within_method"] = (
        df.groupby(["feature_set", "model", "method"])["importance"]
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    return df
