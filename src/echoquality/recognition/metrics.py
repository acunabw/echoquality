from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray, *, min_sensitivity: float = 0.90) -> dict[str, float | str]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    rows: list[tuple[float, float, float, float]] = []
    for threshold in np.linspace(0.01, 0.99, 197):
        pred = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sensitivity = tp / max(1, tp + fn)
        specificity = tn / max(1, tn + fp)
        balanced = 0.5 * (sensitivity + specificity)
        rows.append((float(threshold), float(sensitivity), float(specificity), float(balanced)))
    feasible = [row for row in rows if row[1] >= min_sensitivity]
    if feasible:
        best = max(feasible, key=lambda row: (row[2], row[3], row[0]))
        rule = f"max_specificity_subject_to_sensitivity>={min_sensitivity:.2f}"
    else:
        best = max(rows, key=lambda row: row[3])
        rule = "max_balanced_accuracy_no_feasible_sensitivity_constraint"
    return {
        "threshold": best[0],
        "sensitivity_not_apt": best[1],
        "specificity_apt": best[2],
        "balanced_accuracy": best[3],
        "rule": rule,
    }


def binary_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1_not_apt": float(f1_score(y, pred, zero_division=0)),
        "sensitivity_not_apt": float(tp / max(1, tp + fn)),
        "specificity_apt": float(tn / max(1, tn + fp)),
        "precision_not_apt": float(tp / max(1, tp + fp)),
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if np.unique(y).size == 2 else float("nan"),
        "brier": float(brier_score_loss(y, p)),
    }
