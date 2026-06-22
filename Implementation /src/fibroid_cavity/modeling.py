"""Leakage-aware model training and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib

from fibroid_cavity.plotting import configure_matplotlib

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fibroid_cavity.constants import AUDIT_COLUMNS, GROUP_COLUMN, PREDICTOR_COLUMNS, TARGET_COLUMN


def evaluate_models(
    features: pd.DataFrame,
    output_dir: Path,
    predictors: Optional[list[str]] = None,
    target_column: str = TARGET_COLUMN,
    group_column: str = GROUP_COLUMN,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Train and evaluate classifiers with grouped patient-level CV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    predictors = predictors or PREDICTOR_COLUMNS
    _validate_columns(features, predictors, target_column, group_column)

    data = features.dropna(subset=[target_column, group_column]).copy()
    if data.empty:
        raise ValueError("No usable rows remain after dropping missing target/group values.")
    data[predictors] = data[predictors].replace([np.inf, -np.inf], np.nan)

    x = data[predictors]
    y = data[target_column].astype(int)
    groups = data[group_column].astype(str)

    _write_run_audits(data, predictors, target_column, group_column, output_dir)

    split_count = _safe_split_count(groups, y, n_splits)
    if split_count < 2:
        raise ValueError("At least two patient groups are required for grouped cross-validation.")

    models = build_models(random_state=random_state)
    splitter = StratifiedGroupKFold(n_splits=split_count, shuffle=True, random_state=random_state)

    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []

    for model_name, model in models.items():
        for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            fitted = model.fit(x_train, y_train)
            y_score = _positive_class_score(fitted, x_test)
            y_pred = (y_score >= 0.5).astype(int)
            fold_auc = _safe_auc(y_test, y_score)
            tn, fp, fn, tp = _safe_confusion(y_test, y_pred)

            metrics.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "auc": fold_auc,
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "true_positive": tp,
                    "false_positive": fp,
                    "true_negative": tn,
                    "false_negative": fn,
                }
            )

            fold_predictions = data.iloc[test_idx][[group_column, "fibroid_id"]].copy()
            fold_predictions["model"] = model_name
            fold_predictions["fold"] = fold
            fold_predictions["y_true"] = y_test.to_numpy()
            fold_predictions["y_score"] = y_score
            fold_predictions["y_pred"] = y_pred
            predictions.append(fold_predictions)

        final_model = model.fit(x, y)
        joblib.dump(final_model, output_dir / f"{_slug(model_name)}.joblib")
        _save_feature_importance(final_model, model_name, predictors, output_dir)
        _save_shap_summary(final_model, model_name, predictors, x, output_dir)

    metrics_df = pd.DataFrame(metrics)
    predictions_df = pd.concat(predictions, ignore_index=True)

    metrics_df.to_csv(output_dir / "model_metrics_by_fold.csv", index=False)
    _summarize_metrics(metrics_df).to_csv(output_dir / "model_metrics_summary.csv", index=False)
    predictions_df.to_csv(output_dir / "oof_predictions.csv", index=False)
    _plot_roc_curves(predictions_df, output_dir / "roc_curves.png")

    return metrics_df


def build_models(random_state: int = 42) -> dict[str, Any]:
    """Construct baseline classifiers."""
    models: dict[str, Any] = {
        "Logistic Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

    xgb_classifier = _optional_xgboost(random_state)
    if xgb_classifier is not None:
        models["XGBoost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", xgb_classifier),
            ]
        )

    return models


def _optional_xgboost(random_state: int) -> Optional[Any]:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None

    return XGBClassifier(
        n_estimators=250,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=random_state,
    )


def _validate_columns(
    features: pd.DataFrame,
    predictors: list[str],
    target_column: str,
    group_column: str,
) -> None:
    missing = [column for column in [*predictors, target_column, group_column] if column not in features.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")


def _safe_split_count(groups: pd.Series, y: pd.Series, requested_splits: int) -> int:
    unique_groups = int(groups.nunique())
    class_counts = y.value_counts()
    if class_counts.size < 2:
        raise ValueError("Target must contain both classes for ROC AUC evaluation.")
    min_class_count = int(class_counts.min())
    return min(requested_splits, unique_groups, min_class_count)


def _positive_class_score(model: Any, x_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_test)
        return 1 / (1 + np.exp(-scores))
    return model.predict(x_test).astype(float)


def _safe_auc(y_true: pd.Series, y_score: np.ndarray) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_confusion(y_true: pd.Series, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    labels = [0, 1]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = matrix.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def _summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby("model", as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            folds=("fold", "count"),
        )
        .sort_values("auc_mean", ascending=False)
    )


def _write_run_audits(
    data: pd.DataFrame,
    predictors: list[str],
    target_column: str,
    group_column: str,
    output_dir: Path,
) -> None:
    leakage_audit = pd.DataFrame(
        [
            {
                "column": column,
                "present_in_features": column in data.columns,
                "used_as_predictor": column in predictors,
                "reason": "label derivation / audit only",
            }
            for column in AUDIT_COLUMNS
        ]
    )
    leakage_audit.to_csv(output_dir / "leakage_audit.csv", index=False)

    class_balance = (
        data.groupby(target_column, as_index=False)
        .agg(fibroid_count=(target_column, "size"), patient_count=(group_column, "nunique"))
        .sort_values(target_column)
    )
    class_balance.to_csv(output_dir / "class_balance.csv", index=False)

    numeric_summary = data[predictors].describe().T.reset_index().rename(columns={"index": "feature"})
    numeric_summary.to_csv(output_dir / "feature_summary.csv", index=False)

    config = {
        "predictors": predictors,
        "target_column": target_column,
        "group_column": group_column,
        "excluded_audit_columns": AUDIT_COLUMNS,
        "n_fibroids": int(len(data)),
        "n_patients": int(data[group_column].nunique()),
    }
    (output_dir / "model_run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def _plot_roc_curves(predictions: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(7, 6))

    plotted = False
    for model_name, rows in predictions.groupby("model"):
        if rows["y_true"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(rows["y_true"], rows["y_score"])
        auc = roc_auc_score(rows["y_true"], rows["y_score"])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.2f})")
        plotted = True

    plt.plot([0, 1], [0, 1], color="0.6", linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Grouped CV ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if plotted:
        plt.savefig(output_path, dpi=200)
    plt.close()


def _save_feature_importance(model: Any, model_name: str, predictors: list[str], output_dir: Path) -> None:
    classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else model
    values: Optional[np.ndarray] = None

    if hasattr(classifier, "coef_"):
        values = np.ravel(classifier.coef_)
    elif hasattr(classifier, "feature_importances_"):
        values = np.ravel(classifier.feature_importances_)

    if values is None or len(values) != len(predictors):
        return

    importance = pd.DataFrame({"feature": predictors, "importance": values})
    importance["abs_importance"] = importance["importance"].abs()
    importance = importance.sort_values("abs_importance", ascending=False)
    importance.to_csv(output_dir / f"{_slug(model_name)}_feature_importance.csv", index=False)


def _save_shap_summary(
    model: Any,
    model_name: str,
    predictors: list[str],
    x: pd.DataFrame,
    output_dir: Path,
) -> None:
    try:
        import shap
    except ImportError:
        _write_shap_status(output_dir, "SHAP is not installed. Install the optional interpretability extra to enable SHAP summaries.")
        return

    if x.empty:
        return

    classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else model
    sample = x.sample(n=min(200, len(x)), random_state=42)
    transformed = _transform_pipeline_features(model, sample)

    try:
        explainer = shap.Explainer(classifier, transformed, feature_names=predictors)
        try:
            # Tree explainers can fail a strict additivity check on tiny float
            # tolerances; disable it when the explainer accepts the kwarg.
            values = explainer(transformed, check_additivity=False)
        except TypeError:
            values = explainer(transformed)
    except Exception as exc:  # noqa: BLE001 - SHAP support differs by estimator.
        _write_shap_status(output_dir, f"SHAP failed for {model_name}: {exc}")
        return

    shap_values = np.asarray(values.values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]

    if shap_values.ndim != 2 or shap_values.shape[1] != len(predictors):
        _write_shap_status(output_dir, f"Unexpected SHAP value shape for {model_name}: {shap_values.shape}")
        return

    summary = pd.DataFrame(
        {
            "feature": predictors,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    summary.to_csv(output_dir / f"{_slug(model_name)}_shap_summary.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.barh(summary["feature"], summary["mean_abs_shap"])
    plt.gca().invert_yaxis()
    plt.xlabel("Mean absolute SHAP value")
    plt.title(f"{model_name} SHAP Summary")
    plt.tight_layout()
    plt.savefig(output_dir / f"{_slug(model_name)}_shap_summary.png", dpi=200)
    plt.close()


def _transform_pipeline_features(model: Any, x: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "steps"):
        return x.to_numpy()

    transformed: Any = x
    for step_name, step in model.steps:
        if step_name == "classifier":
            break
        transformed = step.transform(transformed)
    return np.asarray(transformed)


def _write_shap_status(output_dir: Path, message: str) -> None:
    status_path = output_dir / "shap_status.txt"
    existing = status_path.read_text(encoding="utf-8") if status_path.exists() else ""
    if message not in existing:
        status_path.write_text(existing + message + "\n", encoding="utf-8")


def _slug(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")
