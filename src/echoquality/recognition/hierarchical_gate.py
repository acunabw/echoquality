from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from echoquality.constants import LEVELS, feature_level
from echoquality.recognition.fusion import (
    HierarchicalGateBundle,
    fit_late_fusion_gate,
    predict_late_fusion,
)
from echoquality.recognition.metrics import binary_metrics
from echoquality.recognition.models import validated_group_folds


@dataclass
class NestedValidationResult:
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    summary: pd.DataFrame
    selected_features: pd.DataFrame


def infer_feature_columns_by_level(data: pd.DataFrame) -> dict[str, list[str]]:
    result = {level: [] for level in LEVELS}
    for column in data.columns:
        level = feature_level(str(column))
        if level:
            result[level].append(str(column))
    return result


def _summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "roc_auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity_not_apt",
        "specificity_apt",
        "brier",
    ]
    rows: list[dict[str, Any]] = []
    for model_name, group in fold_metrics.groupby("model"):
        row: dict[str, Any] = {"model": model_name, "n_folds": int(group["outer_fold"].nunique())}
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else 0.0
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["roc_auc_mean", "balanced_accuracy_mean", "brier_mean"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    return result


def nested_group_validation(
    data: pd.DataFrame,
    *,
    target_column: str = "not_apt",
    group_column: str = "patient_id",
    model_names: tuple[str, ...] = ("LogisticRegression", "RandomForest"),
    outer_splits: int = 5,
    inner_splits: int = 4,
    seed: int = 2026,
    fast: bool = False,
    min_sensitivity_not_apt: float = 0.90,
) -> NestedValidationResult:
    """Leakage-safe nested patient-group validation of the late-fusion gate."""

    feature_columns = infer_feature_columns_by_level(data)
    for level, columns in feature_columns.items():
        if not columns:
            raise ValueError(f"No {level} feature columns were found")
    y = data[target_column].astype(int).to_numpy()
    groups = data[group_column].astype(str).to_numpy()
    outer_cv = validated_group_folds(y, groups, outer_splits, seed=seed)
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []

    for model_index, model_name in enumerate(model_names):
        for fold_index, (train_index, test_index) in enumerate(outer_cv):
            train = data.iloc[train_index].reset_index(drop=True)
            test = data.iloc[test_index].copy()
            y_train, y_test = y[train_index], y[test_index]
            group_train, group_test = groups[train_index], groups[test_index]
            overlap = set(group_train).intersection(group_test)
            if overlap:
                raise RuntimeError(f"Patient leakage detected: {sorted(overlap)[:5]}")
            bundle, _, _ = fit_late_fusion_gate(
                train,
                y_train,
                group_train,
                feature_columns,
                model_name=model_name,
                inner_splits=inner_splits,
                seed=seed + model_index * 10000 + fold_index * 100,
                fast=fast,
                min_sensitivity_not_apt=min_sensitivity_not_apt,
            )
            predicted = predict_late_fusion(bundle, test)
            metrics = binary_metrics(y_test, predicted["probability_not_apt"].to_numpy(), bundle.threshold)
            metric_rows.append(
                {
                    "model": model_name,
                    "outer_fold": fold_index,
                    "n_train": len(train_index),
                    "n_test": len(test_index),
                    "n_train_patients": len(np.unique(group_train)),
                    "n_test_patients": len(np.unique(group_test)),
                    "patient_overlap": len(overlap),
                    "threshold": bundle.threshold,
                    "threshold_rule": bundle.threshold_rule,
                    **metrics,
                }
            )
            output = test[[column for column in (group_column, "view", "quality", target_column) if column in test]].copy()
            output["model"] = model_name
            output["outer_fold"] = fold_index
            for column in predicted.columns:
                output[column] = predicted[column].to_numpy()
            prediction_rows.append(output)
            for level, fitted in bundle.level_models.items():
                for feature in fitted.selected_features:
                    selected_rows.append(
                        {
                            "model": model_name,
                            "outer_fold": fold_index,
                            "level": level,
                            "feature": feature,
                        }
                    )
    fold_metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    selected_features = pd.DataFrame(selected_rows)
    return NestedValidationResult(
        fold_metrics=fold_metrics,
        predictions=predictions,
        summary=_summary(fold_metrics),
        selected_features=selected_features,
    )


def fit_final_gate(
    data: pd.DataFrame,
    *,
    model_name: str,
    target_column: str = "not_apt",
    group_column: str = "patient_id",
    inner_splits: int = 4,
    seed: int = 2026,
    fast: bool = False,
    min_sensitivity_not_apt: float = 0.90,
) -> HierarchicalGateBundle:
    columns = infer_feature_columns_by_level(data)
    return fit_late_fusion_gate(
        data,
        data[target_column].astype(int).to_numpy(),
        data[group_column].astype(str).to_numpy(),
        columns,
        model_name=model_name,
        inner_splits=inner_splits,
        seed=seed,
        fast=fast,
        min_sensitivity_not_apt=min_sensitivity_not_apt,
    )[0]


def save_gate(bundle: HierarchicalGateBundle, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    metadata = {
        "model_family": bundle.model_family,
        "threshold": bundle.threshold,
        "threshold_rule": bundle.threshold_rule,
        "metadata": bundle.metadata,
        "levels": {
            level: {
                "input_features": fitted.feature_names,
                "selected_features": fitted.selected_features,
                "best_params": fitted.best_params,
            }
            for level, fitted in bundle.level_models.items()
        },
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_gate(path: str | Path) -> HierarchicalGateBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, HierarchicalGateBundle):
        raise TypeError(f"Unexpected artifact type: {type(bundle)}")
    return bundle
