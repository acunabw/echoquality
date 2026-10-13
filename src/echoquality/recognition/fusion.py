from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from echoquality.constants import LEVELS
from echoquality.recognition.models import (
    base_model_spec,
    k_values,
    meta_model_spec,
    safe_group_splits,
    validated_group_folds,
    selected_feature_names,
)


@dataclass
class FittedLevelModel:
    level: str
    feature_names: list[str]
    estimator: Any
    selected_features: list[str]
    best_params: dict[str, Any]


@dataclass
class HierarchicalGateBundle:
    model_family: str
    level_models: dict[str, FittedLevelModel]
    meta_model: Any
    threshold: float
    threshold_rule: str
    feature_columns_by_level: dict[str, list[str]]
    metadata: dict[str, Any]


def _nonconstant_columns(data: pd.DataFrame, columns: list[str]) -> list[str]:
    result: list[str] = []
    for column in columns:
        values = pd.to_numeric(data[column], errors="coerce")
        if values.notna().sum() >= 2 and values.nunique(dropna=True) > 1:
            result.append(column)
    return result


def _grid_search(
    estimator: Any,
    grid: dict[str, list[Any]],
    x: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    requested_splits: int,
    seed: int,
) -> GridSearchCV:
    cv = validated_group_folds(y, groups, requested_splits, seed=seed)
    search = GridSearchCV(
        estimator,
        grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(x, y, groups=groups)
    return search


def _stacking_oof_for_level(
    data: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    model_name: str,
    *,
    requested_splits: int,
    seed: int,
    fast: bool,
) -> tuple[np.ndarray, Any, list[str], dict[str, Any]]:
    """Tune inside the current outer-training set, then create group-OOF scores.

    Hyperparameters are selected by inner patient-group CV.  The selected, fixed
    pipeline is then cloned across the same type of patient-group folds to
    generate out-of-fold scores for the late-fusion meta learner.  The untouched
    outer test fold remains fully isolated, preserving nested-CV evaluation while
    avoiding an unnecessary third CV layer.
    """

    usable = _nonconstant_columns(data, feature_names)
    if not usable:
        raise ValueError(f"No nonconstant features are available for a representation level: {feature_names[:3]}")
    x = data[usable].apply(pd.to_numeric, errors="coerce")
    split_count = safe_group_splits(y, groups, requested_splits)
    n_patients = len(np.unique(groups))
    estimator, grid = base_model_spec(
        model_name,
        k_values(n_patients, len(usable), fast=fast),
        seed,
        fast=fast,
    )
    final_search = _grid_search(
        estimator,
        grid,
        x,
        y,
        groups,
        requested_splits=split_count,
        seed=seed + 500,
    )
    best = final_search.best_estimator_
    stack_cv = validated_group_folds(y, groups, split_count, seed=seed + 1)
    oof = np.full(len(data), np.nan, dtype=float)
    for train_index, validation_index in stack_cv:
        fitted = clone(best)
        fitted.fit(x.iloc[train_index], y[train_index])
        oof[validation_index] = fitted.predict_proba(x.iloc[validation_index])[:, 1]
    if not np.isfinite(oof).all():
        raise RuntimeError("Failed to generate complete out-of-fold level scores")
    selected = selected_feature_names(best, usable)
    return oof, best, selected, dict(final_search.best_params_)

def _meta_oof_and_fit(
    scores: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    *,
    requested_splits: int,
    seed: int,
    fast: bool,
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    split_count = safe_group_splits(y, groups, requested_splits)
    estimator, grid = meta_model_spec(model_name, seed, fast=fast)
    final_search = _grid_search(
        estimator,
        grid,
        scores,
        y,
        groups,
        requested_splits=split_count,
        seed=seed + 800,
    )
    best = final_search.best_estimator_
    cv = validated_group_folds(y, groups, split_count, seed=seed + 1)
    oof = np.full(len(scores), np.nan, dtype=float)
    for train_index, validation_index in cv:
        fitted = clone(best)
        fitted.fit(scores.iloc[train_index], y[train_index])
        oof[validation_index] = fitted.predict_proba(scores.iloc[validation_index])[:, 1]
    if not np.isfinite(oof).all():
        raise RuntimeError("Failed to generate complete out-of-fold meta scores")
    return oof, best, dict(final_search.best_params_)

def fit_late_fusion_gate(
    data: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    feature_columns_by_level: dict[str, list[str]],
    *,
    model_name: str,
    inner_splits: int = 4,
    seed: int = 2026,
    fast: bool = False,
    min_sensitivity_not_apt: float = 0.90,
) -> tuple[HierarchicalGateBundle, pd.DataFrame, np.ndarray]:
    """Fit independent level models and an out-of-fold late-fusion meta model."""

    from echoquality.recognition.metrics import choose_threshold

    level_scores: dict[str, np.ndarray] = {}
    level_models: dict[str, FittedLevelModel] = {}
    for level_index, level in enumerate(LEVELS):
        columns = feature_columns_by_level.get(level, [])
        if not columns:
            raise ValueError(f"No features configured for level {level}")
        oof, estimator, selected, params = _stacking_oof_for_level(
            data,
            y,
            groups,
            columns,
            model_name,
            requested_splits=inner_splits,
            seed=seed + level_index * 1000,
            fast=fast,
        )
        usable = _nonconstant_columns(data, columns)
        level_scores[level] = oof
        level_models[level] = FittedLevelModel(
            level=level,
            feature_names=usable,
            estimator=estimator,
            selected_features=selected,
            best_params=params,
        )
    score_frame = pd.DataFrame({f"score_{level}": level_scores[level] for level in LEVELS})
    meta_oof, meta_model, meta_params = _meta_oof_and_fit(
        score_frame,
        y,
        groups,
        model_name,
        requested_splits=inner_splits,
        seed=seed + 9000,
        fast=fast,
    )
    threshold_info = choose_threshold(y, meta_oof, min_sensitivity=min_sensitivity_not_apt)
    bundle = HierarchicalGateBundle(
        model_family=model_name,
        level_models=level_models,
        meta_model=meta_model,
        threshold=float(threshold_info["threshold"]),
        threshold_rule=str(threshold_info["rule"]),
        feature_columns_by_level={level: model.feature_names for level, model in level_models.items()},
        metadata={
            "inner_splits": inner_splits,
            "seed": seed,
            "fast": fast,
            "min_sensitivity_not_apt": min_sensitivity_not_apt,
            "meta_best_params": meta_params,
            "threshold_training_metrics": threshold_info,
            "feature_selection_scope": "inside_training_folds_only",
            "fusion": "late_fusion_three_independent_representation_levels",
        },
    )
    return bundle, score_frame, meta_oof


def predict_late_fusion(bundle: HierarchicalGateBundle, data: pd.DataFrame) -> pd.DataFrame:
    scores: dict[str, np.ndarray] = {}
    for level in LEVELS:
        fitted = bundle.level_models[level]
        x = data[fitted.feature_names].apply(pd.to_numeric, errors="coerce")
        scores[f"score_{level}"] = fitted.estimator.predict_proba(x)[:, 1]
    score_frame = pd.DataFrame(scores, index=data.index)
    probability = bundle.meta_model.predict_proba(score_frame)[:, 1]
    result = score_frame.copy()
    result["probability_not_apt"] = probability
    result["predicted_not_apt"] = (probability >= bundle.threshold).astype(int)
    result["dominant_level"] = result[[f"score_{level}" for level in LEVELS]].idxmax(axis=1).str.replace("score_", "", regex=False)
    return result
