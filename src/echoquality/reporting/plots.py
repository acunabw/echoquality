from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix


def save_roc(predictions: pd.DataFrame, output: str | Path, *, model_name: str) -> Path:
    selected = predictions[predictions["model"].astype(str) == model_name]
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_predictions(
        selected["not_apt"].astype(int),
        selected["probability_not_apt"].astype(float),
        name=model_name,
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_title("Hierarchical Quality Gate ROC")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_confusion(predictions: pd.DataFrame, output: str | Path, *, model_name: str) -> Path:
    selected = predictions[predictions["model"].astype(str) == model_name]
    matrix = confusion_matrix(
        selected["not_apt"].astype(int),
        selected["predicted_not_apt"].astype(int),
        labels=[0, 1],
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(matrix, display_labels=["APTA", "NO APTA"])
    display.plot(ax=ax, cmap="viridis", colorbar=False)
    ax.set_title("Grouped out-of-fold confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_level_score_boxplot(predictions: pd.DataFrame, output: str | Path, *, model_name: str) -> Path:
    selected = predictions[predictions["model"].astype(str) == model_name]
    columns = ["score_global", "score_anatomical", "score_textural"]
    apt = [selected.loc[selected["not_apt"] == 0, column].to_numpy() for column in columns]
    not_apt = [selected.loc[selected["not_apt"] == 1, column].to_numpy() for column in columns]
    positions_apt = np.array([1, 3, 5], dtype=float)
    positions_poor = positions_apt + 0.7
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot(apt, positions=positions_apt, widths=0.55, patch_artist=False)
    ax.boxplot(not_apt, positions=positions_poor, widths=0.55, patch_artist=False)
    ax.set_xticks(positions_apt + 0.35, ["Global", "Anatomical", "Textural"])
    ax.set_ylabel("Probability of NO APTA")
    ax.set_title("Level-specific out-of-fold scores (paired boxes: APTA, NO APTA)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
