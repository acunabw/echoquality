from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return output


def experiment_summary(
    *,
    experiment_name: str,
    manifest: pd.DataFrame,
    view_features: pd.DataFrame,
    model_summary: pd.DataFrame,
    indicator_tables: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    best = model_summary.iloc[0].to_dict() if not model_summary.empty else {}
    indicators: dict[str, int] = {}
    if indicator_tables:
        for level, table in indicator_tables.items():
            indicators[level] = int(len(table))
    return {
        "experiment": experiment_name,
        "n_patients": int(manifest["patient_id"].nunique()),
        "n_frames": int(len(manifest)),
        "n_patient_views": int(len(view_features)),
        "quality_counts_frames": manifest["quality"].value_counts().to_dict(),
        "quality_counts_patient_view": view_features["quality"].value_counts().to_dict(),
        "exploratory_indicators_by_level": indicators,
        "best_model": best,
        "methodology": {
            "architecture": "three-level hierarchical late fusion",
            "levels": ["global", "anatomical", "textural"],
            "grouping": "patient_id",
            "feature_selection": "inside nested cross-validation",
            "models_preserved": ["LogisticRegression", "RandomForest"],
        },
    }


def compare_experiment_summaries(paths: list[str | Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        best = payload.get("best_model", {})
        rows.append(
            {
                "experiment": payload.get("experiment"),
                "n_patients": payload.get("n_patients"),
                "n_frames": payload.get("n_frames"),
                "n_patient_views": payload.get("n_patient_views"),
                "best_model": best.get("model"),
                "roc_auc_mean": best.get("roc_auc_mean"),
                "balanced_accuracy_mean": best.get("balanced_accuracy_mean"),
                "sensitivity_not_apt_mean": best.get("sensitivity_not_apt_mean"),
                "specificity_apt_mean": best.get("specificity_apt_mean"),
                "brier_mean": best.get("brier_mean"),
            }
        )
    return pd.DataFrame(rows).sort_values("n_patients").reset_index(drop=True)
