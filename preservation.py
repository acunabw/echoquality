from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from skimage import feature, filters, measure, morphology
from skimage.metrics import structural_similarity

from .features import estimate_support, extract_features
from .image_io import robust_scale


def dice_score(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=bool)
    bb = np.asarray(b, dtype=bool)
    denom = int(aa.sum() + bb.sum())
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(aa, bb).sum() / denom)


def _largest_component(binary: np.ndarray) -> np.ndarray:
    lab = measure.label(binary, connectivity=2)
    if lab.max() == 0:
        return np.zeros_like(binary, dtype=bool)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == int(np.argmax(counts))


def proxy_segment_lv(image: np.ndarray, anatomical_mask: np.ndarray | None = None) -> np.ndarray:
    """Segmentador proxy para la demostracion.

    No reemplaza una U-Net. Solo permite mostrar como medir Dice antes/despues
    cuando existe una mascara de referencia. En el experimento real, esta funcion
    debe sustituirse por el segmentador del proyecto de tesis.
    """

    x = robust_scale(image)
    support = estimate_support(x)
    if anatomical_mask is not None and np.any(anatomical_mask > 0):
        roi = morphology.dilation(np.asarray(anatomical_mask) > 0, morphology.disk(5))
        support &= roi
    values = x[support]
    if values.size < 20:
        return np.zeros_like(x, dtype=bool)
    try:
        threshold = float(filters.threshold_otsu(values))
    except ValueError:
        threshold = float(np.percentile(values, 30))
    cavity = support & (x <= min(threshold, float(np.percentile(values, 42))))
    cavity = morphology.opening(cavity, morphology.disk(2))
    cavity = morphology.closing(cavity, morphology.disk(4))
    cavity = morphology.remove_small_objects(cavity, max_size=max(19, x.size // 1000 - 1))
    if anatomical_mask is not None and np.any(anatomical_mask == 1):
        # Selecciona el componente con mayor superposicion con el LV de referencia.
        lab = measure.label(cavity, connectivity=2)
        target = np.asarray(anatomical_mask) == 1
        best = np.zeros_like(cavity, dtype=bool)
        best_overlap = -1
        for label_id in range(1, lab.max() + 1):
            component = lab == label_id
            overlap = int(np.logical_and(component, target).sum())
            if overlap > best_overlap:
                best, best_overlap = component, overlap
        return best
    return _largest_component(cavity)


def edge_f1(original: np.ndarray, processed: np.ndarray) -> float:
    a = feature.canny(robust_scale(original), sigma=1.0)
    b = feature.canny(robust_scale(processed), sigma=1.0)
    # Tolerancia de un pixel para no penalizar pequenas variaciones de gradiente.
    a_d = morphology.dilation(a, morphology.disk(1))
    b_d = morphology.dilation(b, morphology.disk(1))
    precision = np.logical_and(b, a_d).sum() / max(1, b.sum())
    recall = np.logical_and(a, b_d).sum() / max(1, a.sum())
    return float(2 * precision * recall / max(1e-12, precision + recall))


def texture_drift(
    original_features: dict[str, float],
    processed_features: dict[str, float],
    feature_names: Iterable[str],
    scales: dict[str, float] | None = None,
) -> float:
    diffs = []
    for name in feature_names:
        a = float(original_features.get(name, np.nan))
        b = float(processed_features.get(name, np.nan))
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        scale = float(scales.get(name, 0.0)) if scales else 0.0
        denom = max(abs(a), scale, 1e-6)
        diffs.append(abs(b - a) / denom)
    return float(np.median(diffs)) if diffs else 0.0


def preservation_metrics(
    original: np.ndarray,
    processed: np.ndarray,
    *,
    mask: np.ndarray | None = None,
    original_features: dict[str, float] | None = None,
    processed_features: dict[str, float] | None = None,
    protected_texture_features: Iterable[str] = (),
    scales: dict[str, float] | None = None,
) -> dict[str, float]:
    a = robust_scale(original)
    b = robust_scale(processed)
    if a.shape != b.shape:
        raise ValueError("Las transformaciones candidatas no deben cambiar la geometria en esta etapa")
    support = estimate_support(a)
    data_range = 1.0
    ssim = float(structural_similarity(a, b, data_range=data_range))
    grad_a = filters.sobel(a)
    grad_b = filters.sobel(b)
    aa, bb = grad_a[support], grad_b[support]
    if aa.size > 2 and np.std(aa) > 1e-12 and np.std(bb) > 1e-12:
        gradient_corr = float(np.corrcoef(aa, bb)[0, 1])
    else:
        gradient_corr = 0.0
    result = {
        "ssim_original": ssim,
        "edge_f1_original": edge_f1(a, b),
        "gradient_correlation": gradient_corr,
        "clipping_fraction": float(((b <= 0.005) | (b >= 0.995))[support].mean()),
    }
    if original_features is None:
        original_features = extract_features(a, mask)
    if processed_features is None:
        processed_features = extract_features(b, mask)
    result["texture_drift"] = texture_drift(
        original_features, processed_features, protected_texture_features, scales=scales
    )
    if mask is not None and np.any(mask == 1):
        truth = np.asarray(mask) == 1
        pred_before = proxy_segment_lv(a, mask)
        pred_after = proxy_segment_lv(b, mask)
        dice_before = dice_score(truth, pred_before)
        dice_after = dice_score(truth, pred_after)
        result.update(
            {
                "dice_before_proxy": dice_before,
                "dice_after_proxy": dice_after,
                "dice_gain_proxy": dice_after - dice_before,
            }
        )
    else:
        result.update({"dice_before_proxy": float("nan"), "dice_after_proxy": float("nan"), "dice_gain_proxy": 0.0})
    return result
