from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def choose_threshold(
    y_true: pd.Series | np.ndarray,
    proba_not_apt: np.ndarray,
    *,
    min_sensitivity_not_apt: float = 0.90,
) -> dict[str, float | str]:
    """Escoge un umbral que prioriza detectar imagenes no aptas."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba_not_apt, dtype=float)
    rows = []
    for threshold in np.linspace(0.02, 0.98, 193):
        pred = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sensitivity = tp / max(1, tp + fn)
        specificity = tn / max(1, tn + fp)
        balanced = 0.5 * (sensitivity + specificity)
        rows.append((threshold, sensitivity, specificity, balanced))
    feasible = [r for r in rows if r[1] >= min_sensitivity_not_apt]
    if feasible:
        best = max(feasible, key=lambda r: (r[2], r[3], r[0]))
        rule = f"max_especificidad_con_sensibilidad>={min_sensitivity_not_apt:.2f}"
    else:
        best = max(rows, key=lambda r: r[3])
        rule = "max_balanced_accuracy_sin_restriccion_factible"
    return {
        "threshold": float(best[0]),
        "sensitivity_not_apt": float(best[1]),
        "specificity_apt": float(best[2]),
        "balanced_accuracy": float(best[3]),
        "rule": rule,
    }


def binary_metrics(y_true: pd.Series | np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_not_apt": float(f1_score(y, pred, zero_division=0)),
        "sensitivity_not_apt": float(tp / max(1, tp + fn)),
        "specificity_apt": float(tn / max(1, tn + fp)),
        "precision_not_apt": float(tp / max(1, tp + fp)),
        "brier": float(brier_score_loss(y, p)),
    }
    metrics["roc_auc"] = float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")
    metrics["pr_auc"] = float(average_precision_score(y, p)) if np.unique(y).size == 2 else float("nan")
    return metrics


def aggregate_patient_view(data: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Agrega ED/ES por mediana para que la etiqueta de calidad no se duplique."""

    required = {"patient_id", "view", "quality"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Faltan columnas para agregar: {sorted(missing)}")
    numeric = data[feature_columns].apply(pd.to_numeric, errors="coerce")
    temp = pd.concat([data[["patient_id", "view", "quality"]].reset_index(drop=True), numeric], axis=1)
    grouped = temp.groupby(["patient_id", "view", "quality"], as_index=False)[feature_columns].median()
    grouped["not_apt"] = (grouped["quality"] == "Poor").astype(int)
    grouped["apt_target"] = 1 - grouped["not_apt"]
    return grouped
