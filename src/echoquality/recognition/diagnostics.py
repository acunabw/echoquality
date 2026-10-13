from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from echoquality.constants import LEVELS
from echoquality.recognition.fusion import HierarchicalGateBundle, predict_late_fusion
from echoquality.recognition.good_reference import GoodReferenceBundle
from echoquality.recognition.quality_levels import QualityDecision, operational_status


@dataclass
class LevelDiagnosis:
    level: str
    probability_not_apt: float
    mahalanobis_distance: float
    reference_threshold_q95: float
    reference_alarm: bool
    top_deviations: list[dict[str, Any]]
    issue_categories: list[str]


@dataclass
class HierarchicalDiagnosis:
    decision: QualityDecision
    levels: dict[str, LevelDiagnosis]
    dominant_issue: str | None
    processing_families: list[str]


def _issue_category(feature: str) -> str:
    name = feature.lower()
    if "boundary" in name or "cnr" in name:
        return "boundary_visibility"
    if any(token in name for token in ("circularity", "compactness", "eccentricity", "solidity", "aspect", "perimeter", "thickness", "component", "border_touch")):
        return "anatomical_geometry"
    if "dropout" in name:
        return "dropout"
    if "saturation" in name:
        return "saturation"
    if "speckle" in name or "noise" in name or "high_frequency" in name:
        return "noise_or_speckle"
    if "edge" in name or "tenengrad" in name or "laplacian" in name:
        return "sharpness_or_edges"
    if any(token in name for token in ("brightness", "intensity", "range", "contrast")):
        return "brightness_or_contrast"
    if any(token in name for token in ("glcm", "glrlm", "glszm", "gldm", "ngtdm", "lbp", "entropy")):
        return "texture"
    return "other"


def _processing_family(issue: str) -> str | None:
    mapping = {
        "brightness_or_contrast": "global_contrast",
        "saturation": "global_dynamic_compression_review",
        "noise_or_speckle": "texture_preserving_denoising",
        "sharpness_or_edges": "conservative_sharpening",
        "boundary_visibility": "boundary_enhancement",
        "texture": "texture_preserving_denoising",
        "dropout": "review_or_reacquisition",
        "anatomical_geometry": "review_or_reacquisition",
    }
    return mapping.get(issue)


def diagnose_one(
    row: pd.Series | dict[str, Any],
    prediction: pd.Series | dict[str, Any],
    references: GoodReferenceBundle,
    gate: HierarchicalGateBundle,
    *,
    top_n: int = 5,
    conditional_margin: float = 0.15,
) -> HierarchicalDiagnosis:
    view = str(row.get("view", "ALL"))
    levels: dict[str, LevelDiagnosis] = {}
    any_reference_alarm = False
    all_issues: list[tuple[str, float]] = []
    for level in LEVELS:
        reference = references.get(level, view)
        z = reference.robust_z(row)
        ordered = sorted(z.items(), key=lambda item: abs(item[1]), reverse=True)[:top_n]
        top = [
            {
                "feature": feature,
                "robust_z": float(value),
                "direction": "above_good" if value > 0 else "below_good",
                "issue": _issue_category(feature),
            }
            for feature, value in ordered
        ]
        distance = reference.mahalanobis(row)
        alarm = bool(distance > reference.distance_q95)
        any_reference_alarm = any_reference_alarm or alarm
        issues = list(dict.fromkeys(item["issue"] for item in top if abs(float(item["robust_z"])) >= 2.0))
        for item in top:
            magnitude = abs(float(item["robust_z"]))
            if magnitude >= 2.0:
                all_issues.append((str(item["issue"]), magnitude))
        levels[level] = LevelDiagnosis(
            level=level,
            probability_not_apt=float(prediction[f"score_{level}"]),
            mahalanobis_distance=distance,
            reference_threshold_q95=reference.distance_q95,
            reference_alarm=alarm,
            top_deviations=top,
            issue_categories=issues,
        )
    level_probabilities = {level: levels[level].probability_not_apt for level in LEVELS}
    decision = operational_status(
        float(prediction["probability_not_apt"]),
        gate.threshold,
        level_probabilities,
        conditional_margin=conditional_margin,
        reference_alarm=any_reference_alarm,
    )
    actionable_issues = [item for item in all_issues if item[0] != "other"]
    ranked_issues = actionable_issues or all_issues
    dominant_issue = max(ranked_issues, key=lambda item: item[1])[0] if ranked_issues else None
    families = []
    for issue, _ in sorted(all_issues, key=lambda item: item[1], reverse=True):
        family = _processing_family(issue)
        if family and family not in families:
            families.append(family)
    return HierarchicalDiagnosis(decision, levels, dominant_issue, families)


def diagnose_dataset(
    data: pd.DataFrame,
    gate: HierarchicalGateBundle,
    references: GoodReferenceBundle,
    *,
    top_n: int = 5,
    conditional_margin: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = predict_late_fusion(gate, data)
    summary_rows: list[dict[str, Any]] = []
    deviation_rows: list[dict[str, Any]] = []
    for position, (_, row) in enumerate(data.iterrows()):
        prediction = predictions.iloc[position]
        diagnosis = diagnose_one(
            row,
            prediction,
            references,
            gate,
            top_n=top_n,
            conditional_margin=conditional_margin,
        )
        summary = {
            "patient_id": row.get("patient_id"),
            "view": row.get("view"),
            "quality": row.get("quality"),
            "status": diagnosis.decision.status,
            "probability_not_apt": diagnosis.decision.final_probability_not_apt,
            "threshold": diagnosis.decision.threshold,
            "dominant_level": diagnosis.decision.dominant_level,
            "dominant_issue": diagnosis.dominant_issue,
            "processing_families": ";".join(diagnosis.processing_families),
        }
        for level in LEVELS:
            level_diag = diagnosis.levels[level]
            summary[f"score_{level}"] = level_diag.probability_not_apt
            summary[f"distance_{level}"] = level_diag.mahalanobis_distance
            summary[f"reference_alarm_{level}"] = level_diag.reference_alarm
            for item in level_diag.top_deviations:
                deviation_rows.append(
                    {
                        "patient_id": row.get("patient_id"),
                        "view": row.get("view"),
                        "level": level,
                        **item,
                    }
                )
        summary_rows.append(summary)
    return pd.DataFrame(summary_rows), pd.DataFrame(deviation_rows)
