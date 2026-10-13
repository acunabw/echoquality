from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from skimage import feature, filters, morphology
from skimage.metrics import structural_similarity

from echoquality.analysis.extractor import HierarchicalFeatureExtractor
from echoquality.analysis.roi import regions_from_camus_mask
from echoquality.analysis.support import estimate_ultrasound_support
from echoquality.data.io import robust_scale
from echoquality.optimization.objective import ObjectiveWeights, hierarchical_objective
from echoquality.processing.adaptive_processing import candidate_transformations
from echoquality.processing.transforms import TransformationSpec, apply_transformation
from echoquality.recognition.good_reference import GoodReferenceBundle
from echoquality.types import ImageMetadata


@dataclass
class AdaptiveOptimizationResult:
    recommendation: str
    best_spec: TransformationSpec
    best_image: np.ndarray
    candidates: pd.DataFrame
    baseline_features: dict[str, float]
    best_features: dict[str, float]


def _edge_f1(a: np.ndarray, b: np.ndarray) -> float:
    edge_a = feature.canny(a, sigma=1.0)
    edge_b = feature.canny(b, sigma=1.0)
    a_d = morphology.dilation(edge_a, morphology.disk(1))
    b_d = morphology.dilation(edge_b, morphology.disk(1))
    precision = np.logical_and(edge_b, a_d).sum() / max(1, edge_b.sum())
    recall = np.logical_and(edge_a, b_d).sum() / max(1, edge_a.sum())
    return float(2 * precision * recall / max(1e-12, precision + recall))


def _relative_drift(before: dict[str, float], after: dict[str, float], prefix: str) -> float:
    values: list[float] = []
    for name, original in before.items():
        if not name.startswith(prefix) or name not in after:
            continue
        updated = after[name]
        if not np.isfinite(original) or not np.isfinite(updated):
            continue
        values.append(abs(updated - original) / max(abs(original), 1e-6))
    return float(np.median(values)) if values else 0.0


def _reference_distance(
    features: dict[str, float],
    references: GoodReferenceBundle,
    level: str,
    stratum: str,
) -> float:
    reference = references.get(level, stratum)
    return reference.mahalanobis(features)


def optimize_adaptive_processing(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    metadata: ImageMetadata | None,
    references: GoodReferenceBundle,
    reference_stratum: str,
    processing_families: list[str],
    extractor: HierarchicalFeatureExtractor,
    max_candidates: int = 48,
    ssim_min: float = 0.82,
    edge_f1_min: float = 0.70,
    anatomical_drift_max: float = 0.15,
    texture_drift_max: float = 0.20,
    clipping_fraction_max: float = 0.20,
    weights: ObjectiveWeights | None = None,
    min_objective_improvement: float = 0.02,
) -> AdaptiveOptimizationResult:
    weights = weights or ObjectiveWeights()
    original = robust_scale(image, support=estimate_ultrasound_support(image))
    baseline = extractor.extract(original, mask, metadata=metadata).fused()
    before_distance = {
        level: _reference_distance(baseline, references, level, reference_stratum)
        for level in ("global", "anatomical", "textural")
    }
    specs = candidate_transformations(processing_families, max_candidates=max_candidates)
    rows: list[dict[str, Any]] = []
    images: dict[str, np.ndarray] = {}
    feature_sets: dict[str, dict[str, float]] = {}
    for spec in specs:
        processed = apply_transformation(original, spec)
        features = extractor.extract(processed, mask, metadata=metadata).fused()
        after_distance = {
            level: _reference_distance(features, references, level, reference_stratum)
            for level in ("global", "anatomical", "textural")
        }
        gains = {
            level: float((before_distance[level] - after_distance[level]) / max(before_distance[level], 1e-6))
            for level in before_distance
        }
        anatomy_drift = _relative_drift(baseline, features, "anatomical__")
        texture_drift = _relative_drift(baseline, features, "textural__")
        anatomy_preservation = float(np.clip(1.0 - anatomy_drift, -1.0, 1.0))
        texture_component = float(np.clip(gains["textural"] - texture_drift, -1.0, 1.0))
        min_dim = min(original.shape)
        win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
        ssim = float(structural_similarity(original, processed, data_range=1.0, win_size=max(3, win_size)))
        edge_f1 = _edge_f1(original, processed)
        support = estimate_ultrasound_support(original)
        clipping = float(((processed <= 0.005) | (processed >= 0.995))[support].mean())
        valid = bool(
            ssim >= ssim_min
            and edge_f1 >= edge_f1_min
            and anatomy_drift <= anatomical_drift_max
            and texture_drift <= texture_drift_max
            and clipping <= clipping_fraction_max
        )
        objective = hierarchical_objective(
            global_gain=float(np.clip(gains["global"], -1.0, 1.0)),
            anatomical_preservation=anatomy_preservation,
            textural_gain_and_preservation=texture_component,
            ssim=ssim,
            valid=valid,
            weights=weights,
        )
        label = spec.label()
        rows.append(
            {
                "candidate": label,
                "target_level": spec.target_level,
                "rationale": spec.rationale,
                "valid": valid,
                "raw_objective": objective,
                "J_G_global_gain": gains["global"],
                "J_M_anatomical_preservation": anatomy_preservation,
                "J_T_textural_gain_minus_drift": texture_component,
                "global_distance_before": before_distance["global"],
                "global_distance_after": after_distance["global"],
                "anatomical_distance_before": before_distance["anatomical"],
                "anatomical_distance_after": after_distance["anatomical"],
                "textural_distance_before": before_distance["textural"],
                "textural_distance_after": after_distance["textural"],
                "anatomical_drift": anatomy_drift,
                "texture_drift": texture_drift,
                "ssim": ssim,
                "edge_f1": edge_f1,
                "clipping_fraction": clipping,
            }
        )
        images[label] = processed
        feature_sets[label] = features
    table = pd.DataFrame(rows)
    identity_raw = float(table.loc[table["candidate"] == "identity", "raw_objective"].iloc[0])
    table["objective"] = table["raw_objective"] - identity_raw
    table = table.sort_values(["valid", "objective"], ascending=[False, False]).reset_index(drop=True)
    identity = next(spec for spec in specs if spec.label() == "identity")
    valid_table = table[table["valid"]]
    if valid_table.empty or float(valid_table.iloc[0]["objective"]) < float(min_objective_improvement):
        best = identity
        recommendation = "MANTENER_ORIGINAL"
    else:
        label = str(valid_table.iloc[0]["candidate"])
        best = next(spec for spec in specs if spec.label() == label)
        recommendation = "MANTENER_ORIGINAL" if label == "identity" else f"APLICAR_{best.name.upper()}"
    label = best.label()
    return AdaptiveOptimizationResult(
        recommendation=recommendation,
        best_spec=best,
        best_image=images.get(label, original),
        candidates=table,
        baseline_features=baseline,
        best_features=feature_sets.get(label, baseline),
    )
