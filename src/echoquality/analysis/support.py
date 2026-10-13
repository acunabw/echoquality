from __future__ import annotations

import numpy as np
from skimage import measure, morphology


def largest_component(binary: np.ndarray) -> np.ndarray:
    mask = np.asarray(binary, dtype=bool)
    labels = measure.label(mask, connectivity=2)
    if labels.max() == 0:
        return mask
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def estimate_ultrasound_support(image: np.ndarray) -> np.ndarray:
    """Estimate the acquisition fan/support while excluding black padding.

    The rule is intentionally conservative and is recorded as a configurable
    engineering heuristic rather than a physical property of the scanner.
    """

    x = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=bool)
    finite_values = x[finite]
    lo, hi = np.percentile(finite_values, [1, 99])
    normalized = np.zeros_like(x, dtype=float)
    if hi > lo:
        normalized[finite] = np.clip((x[finite] - lo) / (hi - lo), 0.0, 1.0)
    threshold = max(0.01, float(np.percentile(normalized[finite], 30)) * 0.25)
    support = finite & (normalized > threshold)
    support = morphology.closing(support, morphology.disk(3))
    # Explicit connected-component size filter. This implements the same
    # minimum-area criterion without depending on scikit-image's renamed
    # remove_small_objects parameter (min_size -> max_size).
    min_area = max(64, x.size // 500)
    # Connectivity=1 matches the default used by remove_small_objects.
    labels = measure.label(support, connectivity=1)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    support = support & (sizes[labels] >= min_area)
    support = largest_component(support)
    if support.mean() < 0.05:
        support = finite.copy()
    return support


def safe_values(image: np.ndarray, region: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)[np.asarray(region, dtype=bool)]
    values = values[np.isfinite(values)]
    return values if values.size else np.array([0.0], dtype=float)
