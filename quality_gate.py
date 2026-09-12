from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import binary_metrics, choose_threshold


@dataclass
class QualityGateCVResult:
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    summary: pd.DataFrame
    selected_features: pd.DataFrame


def _safe_splits(y: pd.Series, groups: pd.Series, requested: int) -> int:
    group_table = pd.DataFrame({"y": y.to_numpy(), "g": groups.astype(str).to_numpy()}).drop_duplicates("g")
    min_groups = int(group_table.groupby("y")["g"].nunique().min())
    return max(2, min(requested, min_groups))


def _feature_cap(n_train_patients: int, n_features: int) -> int:
    return max(2, min(n_features, 20, int(np.floor(0.30 * n_train_patients))))


def _specs(k_values: list[int], seed: int) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    logistic = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_classif)),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=seed,
                ),
            ),
        ]
    )
    logistic_grid = {
        "selector__k": k_values,
        "model__C": [0.1, 0.5, 1.0, 5.0],
    }
    forest = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", SelectKBest(score_func=f_classif)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )
    forest_grid = {
        "selector__k": k_values,
        "model__max_depth": [4, 8, None],
        "model__min_samples_leaf": [2, 4],
        "model__max_features": ["sqrt", 0.6],
    }
    return {"LogisticRegression": (logistic, logistic_grid), "RandomForest": (forest, forest_grid)}


def _selected_names(estimator: BaseEstimator, feature_names: list[str]) -> list[str]:
    try:
        selector = estimator.named_steps["selector"]  # type: ignore[attr-defined]
        support = selector.get_support()
        return list(np.asarray(feature_names)[support])
    except Exception:
        return []


def nested_group_validation(
    data: pd.DataFrame,
    feature_columns: list[str],
    *,
    target_column: str = "not_apt",
    group_column: str = "patient_id",
    outer_splits: int = 5,
    inner_splits: int = 4,
    min_sensitivity_not_apt: float = 0.90,
    seed: int = 2026,
    model_names: tuple[str, ...] = ("LogisticRegression", "RandomForest"),
) -> QualityGateCVResult:
    """Validacion anidada y agrupada por paciente para la compuerta de calidad."""

    x = data[feature_columns].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    y = data[target_column].astype(int).reset_index(drop=True)
    groups = data[group_column].astype(str).reset_index(drop=True)
    metadata = data[[c for c in ("patient_id", "view", "quality") if c in data.columns]].reset_index(drop=True)
    outer_n = _safe_splits(y, groups, outer_splits)
    outer = StratifiedGroupKFold(n_splits=outer_n, shuffle=True, random_state=seed)
    fold_rows: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []
    selected_rows: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(x, y, groups), start=1):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        g_train, g_test = groups.iloc[train_idx], groups.iloc[test_idx]
        cap = _feature_cap(g_train.nunique(), x.shape[1])
        k_values = sorted(set([min(4, cap), min(8, cap), min(12, cap), cap]))
        inner_n = _safe_splits(y_train, g_train, inner_splits)
        inner = StratifiedGroupKFold(n_splits=inner_n, shuffle=True, random_state=seed + fold)
        available_specs = _specs(k_values, seed + fold)

        for model_name in model_names:
            model, grid = available_specs[model_name]
            search = GridSearchCV(model, grid, scoring="roc_auc", cv=inner, n_jobs=-1, refit=True)
            search.fit(x_train, y_train, groups=g_train)
            best = search.best_estimator_
            inner_oof = cross_val_predict(
                clone(best), x_train, y_train, groups=g_train, cv=inner, method="predict_proba", n_jobs=-1
            )[:, 1]
            threshold_info = choose_threshold(
                y_train, inner_oof, min_sensitivity_not_apt=min_sensitivity_not_apt
            )
            best.fit(x_train, y_train)
            proba = best.predict_proba(x_test)[:, 1]
            metrics = binary_metrics(y_test, proba, float(threshold_info["threshold"]))
            fold_rows.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "n_train_patients": int(g_train.nunique()),
                    "n_test_patients": int(g_test.nunique()),
                    "patient_overlap": int(len(set(g_train) & set(g_test))),
                    "threshold": float(threshold_info["threshold"]),
                    "inner_cv_auc": float(search.best_score_),
                    "feature_cap": cap,
                    "best_params": json.dumps(search.best_params_, ensure_ascii=False),
                    **metrics,
                }
            )
            block = metadata.iloc[test_idx].copy()
            block["fold"] = fold
            block["model"] = model_name
            block["not_apt_true"] = y_test.to_numpy()
            block["proba_not_apt"] = proba
            block["threshold"] = float(threshold_info["threshold"])
            block["not_apt_pred"] = (proba >= float(threshold_info["threshold"])).astype(int)
            pred_rows.append(block)
            for feature_name in _selected_names(best, feature_columns):
                selected_rows.append({"fold": fold, "model": model_name, "feature": feature_name})

    folds = pd.DataFrame(fold_rows)
    summary_rows = []
    for model_name, group in folds.groupby("model"):
        row: dict[str, Any] = {"model": model_name, "n_folds": len(group)}
        for metric in ("roc_auc", "pr_auc", "balanced_accuracy", "sensitivity_not_apt", "specificity_apt", "brier"):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    return QualityGateCVResult(
        folds,
        pd.concat(pred_rows, ignore_index=True),
        summary,
        pd.DataFrame(selected_rows),
    )


def fit_final_quality_gate(
    data: pd.DataFrame,
    feature_columns: list[str],
    *,
    model_name: str = "LogisticRegression",
    target_column: str = "not_apt",
    group_column: str = "patient_id",
    inner_splits: int = 5,
    min_sensitivity_not_apt: float = 0.90,
    review_margin: float = 0.10,
    seed: int = 2026,
) -> dict[str, Any]:
    x = data[feature_columns].apply(pd.to_numeric, errors="coerce")
    y = data[target_column].astype(int)
    groups = data[group_column].astype(str)
    cap = _feature_cap(groups.nunique(), x.shape[1])
    k_values = sorted(set([min(4, cap), min(8, cap), min(12, cap), cap]))
    model, grid = _specs(k_values, seed)[model_name]
    inner_n = _safe_splits(y, groups, inner_splits)
    inner = StratifiedGroupKFold(n_splits=inner_n, shuffle=True, random_state=seed)
    search = GridSearchCV(model, grid, scoring="roc_auc", cv=inner, n_jobs=-1, refit=True)
    search.fit(x, y, groups=groups)
    best = search.best_estimator_
    oof = cross_val_predict(best, x, y, groups=groups, cv=inner, method="predict_proba", n_jobs=-1)[:, 1]
    threshold_info = choose_threshold(y, oof, min_sensitivity_not_apt=min_sensitivity_not_apt)
    best.fit(x, y)
    return {
        "model": best,
        "model_name": model_name,
        "feature_columns": feature_columns,
        "selected_features": _selected_names(best, feature_columns),
        "threshold": float(threshold_info["threshold"]),
        "threshold_rule": str(threshold_info["rule"]),
        "review_margin": float(review_margin),
        "inner_cv_auc": float(search.best_score_),
        "best_params": search.best_params_,
        "positive_class": "not_apt_poor",
        "apt_policy": {"Good": "APTA_DIRECTA", "Medium": "APTA_CONDICIONADA", "Poor": "NO_APTA"},
        "version": "2.0.0",
    }


def predict_quality_gate(bundle: dict[str, Any], feature_row: dict[str, float] | pd.Series) -> dict[str, Any]:
    cols = list(bundle["feature_columns"])
    row = pd.DataFrame([{c: float(feature_row.get(c, np.nan)) for c in cols}])
    proba_not_apt = float(bundle["model"].predict_proba(row)[0, 1])
    threshold = float(bundle["threshold"])
    margin = float(bundle.get("review_margin", 0.10))
    if proba_not_apt >= threshold:
        status = "NO_APTA"
    elif proba_not_apt >= max(0.0, threshold - margin):
        status = "APTA_CONDICIONADA"
    else:
        status = "APTA_DIRECTA"
    return {
        "status": status,
        "proba_not_apt": proba_not_apt,
        "proba_apt": 1.0 - proba_not_apt,
        "threshold": threshold,
        "review_margin": margin,
    }


def save_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, p)
    metadata = {k: v for k, v in bundle.items() if k != "model"}
    p.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def load_bundle(path: str | Path) -> dict[str, Any]:
    return joblib.load(path)
