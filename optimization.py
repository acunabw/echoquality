from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .diagnostics import DiagnosticResult, diagnose_deviations
from .features import extract_features
from .preservation import preservation_metrics
from .reference import GoodReference
from .transforms import TransformationSpec, apply_transformation, candidate_transformations


@dataclass
class OptimizationResult:
    status: str
    diagnostic: DiagnosticResult
    candidates: pd.DataFrame
    best_spec: TransformationSpec
    best_image: np.ndarray
    baseline_features: dict[str, float]
    best_features: dict[str, float]
    recommendation: str


def _relative_gain(after: float, before: float, higher_is_better: bool = True) -> float:
    denom = max(abs(before), 1e-6)
    raw = (after - before) / denom
    return float(raw if higher_is_better else -raw)


def optimize_image(
    image: np.ndarray,
    mask: np.ndarray | None,
    reference: GoodReference,
    *,
    gate_status: str = "APTA_CONDICIONADA",
    deficiency_threshold: float = 0.75,
    max_candidates: int = 48,
    ssim_min: float = 0.82,
    edge_f1_min: float = 0.70,
    texture_drift_max: float = 0.20,
    clipping_max: float = 0.20,
    min_objective_improvement: float = 0.02,
) -> OptimizationResult:
    """Optimiza una imagen apta bajo restricciones de preservacion.

    La funcion no intenta fabricar informacion ausente. Si la compuerta declara
    NO_APTA o el diagnostico sugiere un defecto anatomico severo, recomienda
    repetir adquisicion/revision en lugar de aplicar un filtro agresivo.
    """

    baseline_features = extract_features(image, mask)
    baseline_distance = reference.mahalanobis(baseline_features)
    deviations = reference.deviation_table(baseline_features)
    diagnostic = diagnose_deviations(deviations, deficiency_threshold=deficiency_threshold)
    identity = TransformationSpec("identity", ())
    original01 = apply_transformation(image, identity)

    if gate_status == "NO_APTA" or diagnostic.non_correctable:
        row = {
            "candidate": identity.label(),
            "valid": True,
            "objective": 0.0,
            "good_distance_before": baseline_distance,
            "good_distance_after": baseline_distance,
            "quality_gain": 0.0,
            "reason": "no se procesa: requiere revision o repeticion de adquisicion",
        }
        return OptimizationResult(
            gate_status,
            diagnostic,
            pd.DataFrame([row]),
            identity,
            original01,
            baseline_features,
            baseline_features,
            "REPETIR_ADQUISICION_O_REVISION_EXPERTA",
        )

    specs = candidate_transformations(diagnostic.top_issues, max_candidates=max_candidates)
    if not diagnostic.top_issues:
        specs = [identity]
    protected_texture = [
        name
        for name in reference.feature_names
        if any(token in name.lower() for token in ("glcm", "lbp", "entropy", "local_variance", "speckle"))
    ]
    scale_map = {name: float(reference.scale[i]) for i, name in enumerate(reference.feature_names)}
    rows: list[dict[str, Any]] = []
    images: dict[str, np.ndarray] = {}
    feature_sets: dict[str, dict[str, float]] = {}

    base_contrast = float(np.nanmean([baseline_features.get("local_contrast_mean", np.nan), baseline_features.get("cnr_myo_lv", np.nan)]))
    base_noise = float(np.nanmean([baseline_features.get("noise_mad_highpass", np.nan), baseline_features.get("speckle_index_global", np.nan)]))

    for spec in specs:
        label = spec.label()
        processed = apply_transformation(image, spec)
        features = extract_features(processed, mask)
        distance_after = reference.mahalanobis(features)
        quality_gain = float((baseline_distance - distance_after) / max(baseline_distance, 1e-6))
        contrast_after = float(np.nanmean([features.get("local_contrast_mean", np.nan), features.get("cnr_myo_lv", np.nan)]))
        noise_after = float(np.nanmean([features.get("noise_mad_highpass", np.nan), features.get("speckle_index_global", np.nan)]))
        contrast_gain = _relative_gain(contrast_after, base_contrast, True)
        noise_reduction = _relative_gain(noise_after, base_noise, False)
        preservation = preservation_metrics(
            original01,
            processed,
            mask=mask,
            original_features=baseline_features,
            processed_features=features,
            protected_texture_features=protected_texture,
            scales=scale_map,
        )
        dice_gain = float(preservation.get("dice_gain_proxy", 0.0))
        valid = bool(
            preservation["ssim_original"] >= ssim_min
            and preservation["edge_f1_original"] >= edge_f1_min
            and preservation["texture_drift"] <= texture_drift_max
            and preservation["clipping_fraction"] <= clipping_max
        )
        objective = (
            0.50 * np.clip(quality_gain, -1.0, 1.0)
            + 0.15 * np.clip(contrast_gain, -1.0, 1.0)
            + 0.15 * np.clip(noise_reduction, -1.0, 1.0)
            + 0.15 * np.clip(dice_gain / 0.10, -1.0, 1.0)
            + 0.05 * np.clip((preservation["ssim_original"] - ssim_min) / max(1e-6, 1.0 - ssim_min), -1.0, 1.0)
            - 0.20 * preservation["texture_drift"]
        )
        if not valid:
            objective -= 1.0
        row = {
            "candidate": label,
            "valid": valid,
            "objective": float(objective),
            "good_distance_before": baseline_distance,
            "good_distance_after": distance_after,
            "quality_gain": quality_gain,
            "contrast_gain": contrast_gain,
            "noise_reduction": noise_reduction,
            **preservation,
        }
        rows.append(row)
        images[label] = processed
        feature_sets[label] = features

    table = pd.DataFrame(rows).sort_values(["valid", "objective"], ascending=[False, False]).reset_index(drop=True)
    valid_table = table[table["valid"]]
    if valid_table.empty:
        best_label = identity.label()
        recommendation = "MANTENER_ORIGINAL_NINGUN_CANDIDATO_CUMPLE_RESTRICCIONES"
    else:
        best_label = str(valid_table.iloc[0]["candidate"])
        best_objective = float(valid_table.iloc[0]["objective"])
        if best_objective < min_objective_improvement:
            best_label = identity.label()
            recommendation = "MANTENER_IMAGEN_ORIGINAL"
        else:
            recommendation = f"APLICAR_{best_label}"
    best_spec = next((spec for spec in specs if spec.label() == best_label), identity)
    best_image = images.get(best_label, original01)
    best_features = feature_sets.get(best_label, baseline_features)
    return OptimizationResult(
        gate_status,
        diagnostic,
        table,
        best_spec,
        best_image,
        baseline_features,
        best_features,
        recommendation,
    )
