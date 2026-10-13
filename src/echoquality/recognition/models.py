from __future__ import annotations

from typing import Any
import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold


def safe_f_classif(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fold-local ANOVA scoring that treats constant/non-finite columns as zero.

    ``sklearn.feature_selection.f_classif`` correctly warns when a training fold
    contains a constant feature.  In this project those warnings are expected for
    small or highly stratified folds, so we convert undefined scores to zero while
    retaining the exact SelectKBest/nested-CV design.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=UserWarning)
        scores, pvalues = f_classif(x, y)
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    pvalues = np.nan_to_num(np.asarray(pvalues, dtype=float), nan=1.0, posinf=1.0, neginf=1.0)
    return scores, pvalues


def safe_group_splits(y: np.ndarray, groups: np.ndarray, requested: int) -> int:
    """Cap patient-group folds by the number of groups represented per class.

    This is only a feasibility bound, not a guarantee that the chosen folds
    contain both classes. Use ``validated_group_folds`` to generate folds.
    """
    import pandas as pd

    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups).astype(str)
    if y.ndim != 1 or groups.ndim != 1 or len(y) != len(groups):
        raise ValueError("y and groups must be 1-D arrays of equal length")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Binary quality classification needs both classes 0 and 1")
    table = pd.DataFrame({"y": y, "group": groups})
    # A CAMUS patient may contribute both labels through different views.
    minimum = int(table.groupby("y")["group"].nunique().min())
    if minimum < 2:
        raise ValueError("At least two distinct patient groups per target class are required")
    return max(2, min(int(requested), minimum))


def validated_group_folds(
    y: np.ndarray,
    groups: np.ndarray,
    requested: int,
    *,
    seed: int,
    max_attempts: int = 256,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Find deterministic, leakage-free, binary-valid StratifiedGroupKFold folds.

    StratifiedGroupKFold is heuristic, especially with few minority-class
    patients. A feasible number of groups does NOT ensure each training and
    evaluation split contains both classes. Probe deterministic seeds and
    accept only a complete partition with both classes in each split.

    For AUC-based scoring, both training AND validation need both classes.
    If no such partition exists, fail explicitly instead of fitting a
    single-class classifier or reporting undefined AUC.
    """
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups).astype(str)
    n_splits = safe_group_splits(y, groups, requested)
    for attempt in range(max_attempts):
        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=int(seed) + attempt
        )
        candidate = list(cv.split(np.zeros(len(y)), y, groups))
        if all(
            len(train) > 0
            and len(test) > 0
            and np.unique(y[train]).size == 2
            and np.unique(y[test]).size == 2
            and not set(groups[train]).intersection(groups[test])
            for train, test in candidate
        ):
            return candidate
    counts = {int(label): int(np.sum(y == label)) for label in (0, 1)}
    patient_counts = {
        int(label): int(len(np.unique(groups[y == label]))) for label in (0, 1)
    }
    raise ValueError(
        "Could not construct patient-disjoint, two-class train AND validation "
        f"folds (n_splits={n_splits}, sample_counts={counts}, "
        f"patient_groups_per_class={patient_counts}, attempts={max_attempts}). "
        "Use more independent patients of the minority class, or reduce the "
        "requested folds. Do not duplicate images or mix patient groups."
    )


def feature_cap(n_train_patients: int, n_features: int) -> int:
    return max(1, min(n_features, 24, int(np.floor(0.30 * n_train_patients))))


def k_values(n_train_patients: int, n_features: int, *, fast: bool = False) -> list[int]:
    cap = feature_cap(n_train_patients, n_features)
    if fast:
        return sorted({min(4, cap), cap})
    return sorted({min(3, cap), min(6, cap), min(12, cap), cap})


def base_model_spec(model_name: str, k_options: list[int], seed: int, *, fast: bool = False) -> tuple[Pipeline, dict[str, list[Any]]]:
    if model_name == "LogisticRegression":
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("selector", SelectKBest(score_func=safe_f_classif)),
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
        grid = {
            "selector__k": k_options,
            "model__C": [0.5, 1.0] if fast else [0.1, 0.5, 1.0, 5.0],
        }
        return pipeline, grid
    if model_name == "RandomForest":
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("selector", SelectKBest(score_func=safe_f_classif)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=120 if fast else 300,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        )
        grid = {
            "selector__k": k_options,
            "model__max_depth": [4, None] if fast else [4, 8, None],
            "model__min_samples_leaf": [2, 4],
            "model__max_features": ["sqrt"] if fast else ["sqrt", 0.6],
        }
        return pipeline, grid
    raise ValueError(f"Unsupported model family: {model_name}")


def meta_model_spec(model_name: str, seed: int, *, fast: bool = False) -> tuple[Pipeline, dict[str, list[Any]]]:
    if model_name == "LogisticRegression":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
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
        return model, {"model__C": [0.5, 1.0] if fast else [0.1, 0.5, 1.0, 5.0]}
    if model_name == "RandomForest":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=120 if fast else 300,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=seed,
                    ),
                ),
            ]
        )
        grid = {
            "model__max_depth": [3, None] if fast else [3, 5, None],
            "model__min_samples_leaf": [2, 4],
            "model__max_features": ["sqrt", 1.0],
        }
        return model, grid
    raise ValueError(f"Unsupported model family: {model_name}")


def selected_feature_names(estimator: BaseEstimator, feature_names: list[str]) -> list[str]:
    try:
        selector = estimator.named_steps["selector"]  # type: ignore[attr-defined]
        return list(np.asarray(feature_names)[selector.get_support()])
    except Exception:
        return list(feature_names)
