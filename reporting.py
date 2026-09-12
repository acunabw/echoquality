from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_curve, auc


def save_roc(predictions: pd.DataFrame, path: str | Path, model_name: str | None = None) -> Path:
    data = predictions.copy()
    if model_name is not None and "model" in data:
        data = data[data["model"] == model_name]
    y = data["not_apt_true"].astype(int).to_numpy()
    p = data["proba_not_apt"].astype(float).to_numpy()
    fpr, tpr, _ = roc_curve(y, p)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(fpr, tpr, linewidth=2.4, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Sensibilidad para NO APTA")
    ax.set_title("Curva ROC - compuerta de calidad")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    pth = Path(path)
    pth.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pth, dpi=180)
    plt.close(fig)
    return pth


def save_confusion(predictions: pd.DataFrame, path: str | Path, model_name: str | None = None) -> Path:
    data = predictions.copy()
    if model_name is not None and "model" in data:
        data = data[data["model"] == model_name]
    y = data["not_apt_true"].astype(int).to_numpy()
    pred = data["not_apt_pred"].astype(int).to_numpy()
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    disp = ConfusionMatrixDisplay(cm, display_labels=["APTA", "NO APTA"])
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Matriz de confusion agrupada")
    fig.tight_layout()
    pth = Path(path)
    pth.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pth, dpi=180)
    plt.close(fig)
    return pth


def save_feature_ranking(ranking: pd.DataFrame, path: str | Path, top_n: int = 12) -> Path:
    top = ranking.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.barh(top["feature"], top["association_score"])
    ax.set_xlabel("Puntaje compuesto de asociacion")
    ax.set_title("Indicadores asociados con Good / Medium / Poor")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    pth = Path(path)
    pth.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pth, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return pth


def save_deviation_plot(deviations: pd.DataFrame, path: str | Path, top_n: int = 10) -> Path:
    top = deviations.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    ax.barh(top["feature"], top["deficiency"])
    ax.axvline(0.75, linestyle="--", linewidth=1.2, label="umbral de diagnostico")
    ax.set_xlabel("Desviacion respecto al patron Good")
    ax.set_title("Indicadores responsables de la calidad observada")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    pth = Path(path)
    pth.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pth, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return pth


def save_candidate_scores(candidates: pd.DataFrame, path: str | Path, top_n: int = 12) -> Path:
    top = candidates.sort_values("objective", ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.2, 6.0))
    ax.barh(top["candidate"], top["objective"])
    ax.axvline(0.0, linewidth=1.0)
    ax.set_xlabel("Funcion objetivo")
    ax.set_title("Comparacion de transformaciones candidatas")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    pth = Path(path)
    pth.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pth, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return pth


def save_before_after(original: np.ndarray, processed: np.ndarray, path: str | Path, title: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(processed, cmap="gray")
    axes[1].set_title("Procesada")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    pth = Path(path)
    pth.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pth, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return pth
