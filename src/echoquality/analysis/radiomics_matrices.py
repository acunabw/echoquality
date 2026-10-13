from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

import numpy as np
from scipy import ndimage as ndi
from skimage import measure


def quantize_roi(image01: np.ndarray, mask: np.ndarray, levels: int) -> np.ndarray:
    """Fixed-bin quantization on [0, 1]; outside-ROI pixels receive -1."""

    if levels < 2:
        raise ValueError("levels must be >= 2")
    image = np.asarray(image01, dtype=float)
    roi = np.asarray(mask, dtype=bool)
    quantized = np.full(image.shape, -1, dtype=np.int16)
    quantized[roi] = np.clip(np.floor(image[roi] * levels), 0, levels - 1).astype(np.int16)
    return quantized


def masked_glcm(
    quantized: np.ndarray,
    mask: np.ndarray,
    *,
    levels: int,
    distances: Iterable[int] = (1, 2),
    angles: Iterable[float] = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
    symmetric: bool = True,
) -> np.ndarray:
    q = np.asarray(quantized, dtype=int)
    roi = np.asarray(mask, dtype=bool)
    matrix = np.zeros((levels, levels), dtype=np.float64)
    rows, cols = q.shape
    for distance in distances:
        for angle in angles:
            dy = int(round(np.sin(angle) * distance))
            dx = int(round(np.cos(angle) * distance))
            if dy == 0 and dx == 0:
                continue
            y0a, y1a = max(0, -dy), min(rows, rows - dy)
            x0a, x1a = max(0, -dx), min(cols, cols - dx)
            y0b, y1b = y0a + dy, y1a + dy
            x0b, x1b = x0a + dx, x1a + dx
            valid = roi[y0a:y1a, x0a:x1a] & roi[y0b:y1b, x0b:x1b]
            if not valid.any():
                continue
            first = q[y0a:y1a, x0a:x1a][valid]
            second = q[y0b:y1b, x0b:x1b][valid]
            np.add.at(matrix, (first, second), 1.0)
            if symmetric:
                np.add.at(matrix, (second, first), 1.0)
    total = matrix.sum()
    return matrix / total if total > 0 else matrix


def glcm_features(matrix: np.ndarray) -> dict[str, float]:
    p = np.asarray(matrix, dtype=float)
    levels = p.shape[0]
    i, j = np.indices(p.shape)
    mu_i = float((i * p).sum())
    mu_j = float((j * p).sum())
    sigma_i = float(np.sqrt((((i - mu_i) ** 2) * p).sum()))
    sigma_j = float(np.sqrt((((j - mu_j) ** 2) * p).sum()))
    asm = float((p * p).sum())
    nonzero = p[p > 0]
    return {
        "contrast": float((((i - j) ** 2) * p).sum()),
        "dissimilarity": float((np.abs(i - j) * p).sum()),
        "homogeneity": float((p / (1.0 + (i - j) ** 2)).sum()),
        "correlation": float((((i - mu_i) * (j - mu_j) * p).sum()) / max(sigma_i * sigma_j, 1e-12)),
        "asm_uniformity": asm,
        "energy_sqrt_asm": float(np.sqrt(asm)),
        "entropy": float(-(nonzero * np.log2(nonzero)).sum()) if nonzero.size else 0.0,
    }


def _line_starts(shape: tuple[int, int], dy: int, dx: int) -> list[tuple[int, int]]:
    rows, cols = shape
    starts = []
    for y in range(rows):
        for x in range(cols):
            py, px = y - dy, x - dx
            if not (0 <= py < rows and 0 <= px < cols):
                starts.append((y, x))
    return starts


def glrlm_matrix(
    quantized: np.ndarray,
    mask: np.ndarray,
    *,
    levels: int,
    directions: Iterable[tuple[int, int]] = ((0, 1), (1, 0), (1, 1), (1, -1)),
) -> np.ndarray:
    q = np.asarray(quantized, dtype=int)
    roi = np.asarray(mask, dtype=bool)
    rows, cols = q.shape
    runs: defaultdict[tuple[int, int], float] = defaultdict(float)
    max_run = 1
    for dy, dx in directions:
        for sy, sx in _line_starts(q.shape, dy, dx):
            y, x = sy, sx
            current_gray = -1
            run_length = 0
            while 0 <= y < rows and 0 <= x < cols:
                if roi[y, x]:
                    gray = int(q[y, x])
                    if gray == current_gray:
                        run_length += 1
                    else:
                        if current_gray >= 0 and run_length > 0:
                            runs[(current_gray, run_length)] += 1.0
                            max_run = max(max_run, run_length)
                        current_gray, run_length = gray, 1
                else:
                    if current_gray >= 0 and run_length > 0:
                        runs[(current_gray, run_length)] += 1.0
                        max_run = max(max_run, run_length)
                    current_gray, run_length = -1, 0
                y, x = y + dy, x + dx
            if current_gray >= 0 and run_length > 0:
                runs[(current_gray, run_length)] += 1.0
                max_run = max(max_run, run_length)
    matrix = np.zeros((levels, max_run), dtype=float)
    for (gray, run), count in runs.items():
        matrix[gray, run - 1] += count
    return matrix


def _run_or_zone_features(matrix: np.ndarray, n_pixels: int, *, kind: str) -> dict[str, float]:
    p = np.asarray(matrix, dtype=float)
    total = float(p.sum())
    if total <= 0:
        prefixes = ("sre", "lre", "gln", "rln", "rp", "lgre", "hgre") if kind == "run" else ("sze", "lze", "gln", "zsn", "zp", "lgze", "hgze")
        return {name: 0.0 for name in prefixes}
    gray = np.arange(1, p.shape[0] + 1, dtype=float)[:, None]
    size = np.arange(1, p.shape[1] + 1, dtype=float)[None, :]
    gray_sum = p.sum(axis=1)
    size_sum = p.sum(axis=0)
    if kind == "run":
        return {
            "sre": float((p / (size**2)).sum() / total),
            "lre": float((p * (size**2)).sum() / total),
            "gln": float((gray_sum**2).sum() / total),
            "rln": float((size_sum**2).sum() / total),
            "rp": float(total / max(1, n_pixels)),
            "lgre": float((p / (gray**2)).sum() / total),
            "hgre": float((p * (gray**2)).sum() / total),
        }
    return {
        "sze": float((p / (size**2)).sum() / total),
        "lze": float((p * (size**2)).sum() / total),
        "gln": float((gray_sum**2).sum() / total),
        "zsn": float((size_sum**2).sum() / total),
        "zp": float(total / max(1, n_pixels)),
        "lgze": float((p / (gray**2)).sum() / total),
        "hgze": float((p * (gray**2)).sum() / total),
    }


def glrlm_features(matrix: np.ndarray, n_pixels: int) -> dict[str, float]:
    return _run_or_zone_features(matrix, n_pixels, kind="run")


def glszm_matrix(quantized: np.ndarray, mask: np.ndarray, *, levels: int, connectivity: int = 2) -> np.ndarray:
    q = np.asarray(quantized, dtype=int)
    roi = np.asarray(mask, dtype=bool)
    zones: defaultdict[tuple[int, int], float] = defaultdict(float)
    max_size = 1
    for gray in range(levels):
        labels = measure.label(roi & (q == gray), connectivity=connectivity)
        if labels.max() == 0:
            continue
        counts = np.bincount(labels.ravel())[1:]
        for size in counts[counts > 0]:
            size_int = int(size)
            zones[(gray, size_int)] += 1.0
            max_size = max(max_size, size_int)
    matrix = np.zeros((levels, max_size), dtype=float)
    for (gray, size), count in zones.items():
        matrix[gray, size - 1] += count
    return matrix


def glszm_features(matrix: np.ndarray, n_pixels: int) -> dict[str, float]:
    return _run_or_zone_features(matrix, n_pixels, kind="zone")


def gldm_matrix(
    quantized: np.ndarray,
    mask: np.ndarray,
    *,
    levels: int,
    delta: int = 1,
    alpha: int = 0,
) -> np.ndarray:
    q = np.asarray(quantized, dtype=int)
    roi = np.asarray(mask, dtype=bool)
    dependence = np.ones(q.shape, dtype=np.int16)
    rows, cols = q.shape
    for dy in range(-delta, delta + 1):
        for dx in range(-delta, delta + 1):
            if dy == 0 and dx == 0:
                continue
            y0a, y1a = max(0, -dy), min(rows, rows - dy)
            x0a, x1a = max(0, -dx), min(cols, cols - dx)
            y0b, y1b = y0a + dy, y1a + dy
            x0b, x1b = x0a + dx, x1a + dx
            valid = roi[y0a:y1a, x0a:x1a] & roi[y0b:y1b, x0b:x1b]
            similar = valid & (np.abs(q[y0a:y1a, x0a:x1a] - q[y0b:y1b, x0b:x1b]) <= alpha)
            dependence[y0a:y1a, x0a:x1a] += similar.astype(np.int16)
    max_dep = int(dependence[roi].max()) if roi.any() else 1
    matrix = np.zeros((levels, max_dep), dtype=float)
    np.add.at(matrix, (q[roi], dependence[roi] - 1), 1.0)
    return matrix


def gldm_features(matrix: np.ndarray, n_pixels: int) -> dict[str, float]:
    p = np.asarray(matrix, dtype=float)
    total = float(p.sum())
    if total <= 0:
        return {name: 0.0 for name in ("sde", "lde", "gln", "dn", "dp", "lgle", "hgle")}
    gray = np.arange(1, p.shape[0] + 1, dtype=float)[:, None]
    dependence = np.arange(1, p.shape[1] + 1, dtype=float)[None, :]
    gray_sum = p.sum(axis=1)
    dep_sum = p.sum(axis=0)
    return {
        "sde": float((p / (dependence**2)).sum() / total),
        "lde": float((p * (dependence**2)).sum() / total),
        "gln": float((gray_sum**2).sum() / total),
        "dn": float((dep_sum**2).sum() / total),
        "dp": float(total / max(1, n_pixels)),
        "lgle": float((p / (gray**2)).sum() / total),
        "hgle": float((p * (gray**2)).sum() / total),
    }


def ngtdm_features(quantized: np.ndarray, mask: np.ndarray, *, levels: int, delta: int = 1) -> dict[str, float]:
    q = np.asarray(quantized, dtype=float)
    roi = np.asarray(mask, dtype=bool)
    kernel = np.ones((2 * delta + 1, 2 * delta + 1), dtype=float)
    kernel[delta, delta] = 0.0
    neighbor_count = ndi.convolve(roi.astype(float), kernel, mode="constant", cval=0.0)
    neighbor_sum = ndi.convolve(np.where(roi, q, 0.0), kernel, mode="constant", cval=0.0)
    valid = roi & (neighbor_count > 0)
    if not valid.any():
        return {name: 0.0 for name in ("coarseness", "contrast", "busyness", "complexity", "strength")}
    average = np.zeros_like(q, dtype=float)
    average[valid] = neighbor_sum[valid] / neighbor_count[valid]
    gray_values = np.arange(1, levels + 1, dtype=float)
    n = np.zeros(levels, dtype=float)
    s = np.zeros(levels, dtype=float)
    q_int = quantized.astype(int)
    for gray in range(levels):
        pixels = valid & (q_int == gray)
        n[gray] = pixels.sum()
        s[gray] = np.abs((gray + 1) - (average[pixels] + 1)).sum()
    n_total = float(n.sum())
    p = n / max(n_total, 1.0)
    present = n > 0
    ng = int(present.sum())
    ps = p * s
    coarseness = 1.0 / max(float(ps.sum()), 1e-12)
    pair_diff = (gray_values[:, None] - gray_values[None, :]) ** 2
    probability_pairs = p[:, None] * p[None, :]
    contrast_first = float((probability_pairs * pair_diff).sum() / max(1, ng * (ng - 1))) if ng > 1 else 0.0
    contrast = contrast_first * float(s.sum() / max(n_total, 1.0))
    busyness_denominator = float(np.abs(gray_values[:, None] * p[:, None] - gray_values[None, :] * p[None, :])[present[:, None] & present[None, :]].sum())
    busyness = float(ps.sum() / max(busyness_denominator, 1e-12))
    complexity = 0.0
    strength_numerator = 0.0
    for i in range(levels):
        if not present[i]:
            continue
        for j in range(levels):
            if i == j or not present[j]:
                continue
            complexity += abs((i + 1) - (j + 1)) * (p[i] * s[i] + p[j] * s[j]) / max(n[i] + n[j], 1e-12)
            strength_numerator += (p[i] + p[j]) * ((i + 1) - (j + 1)) ** 2
    strength = float(strength_numerator / max(s.sum(), 1e-12))
    return {
        "coarseness": float(coarseness),
        "contrast": float(contrast),
        "busyness": float(busyness),
        "complexity": float(complexity),
        "strength": float(strength),
    }
