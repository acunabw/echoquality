from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import kurtosis, skew
from skimage import feature, filters, measure, morphology

from .image_io import robust_scale


QUALITY_FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "intensity": (
        "intensity_",
        "dynamic_range",
        "entropy",
        "local_contrast",
        "cnr_",
        "dropout_",
    ),
    "sharpness": ("laplacian_variance", "tenengrad", "edge_", "boundary_"),
    "noise": ("noise_", "speckle_", "high_frequency_"),
    "texture": ("glcm_", "lbp_", "local_variance"),
    "morphology": ("mask_", "lv_", "myo_", "la_"),
}


def _largest_component(binary: np.ndarray) -> np.ndarray:
    labels = measure.label(np.asarray(binary, dtype=bool), connectivity=2)
    if labels.max() == 0:
        return np.asarray(binary, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def estimate_support(image01: np.ndarray) -> np.ndarray:
    """Estima la region de adquisicion y elimina el fondo negro del ecografo."""

    x = np.asarray(image01, dtype=np.float64)
    threshold = max(0.015, float(np.percentile(x, 35)) * 0.25)
    support = x > threshold
    support = morphology.closing(support, morphology.disk(3))
    support = morphology.remove_small_objects(support, max_size=max(63, x.size // 500 - 1))
    support = _largest_component(support)
    if support.mean() < 0.05:
        support = np.ones_like(x, dtype=bool)
    return support


def _safe_values(image: np.ndarray, region: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)[np.asarray(region, dtype=bool)]
    values = values[np.isfinite(values)]
    return values if values.size else np.array([0.0], dtype=np.float64)


def _entropy(values: np.ndarray, bins: int = 64) -> float:
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    p = hist.astype(np.float64)
    if p.sum() <= 0:
        return 0.0
    p /= p.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _binary_region_metrics(binary: np.ndarray, prefix: str) -> dict[str, float]:
    b = np.asarray(binary, dtype=bool)
    total = float(b.size)
    if not b.any():
        return {
            f"{prefix}_fraction": 0.0,
            f"{prefix}_components": 0.0,
            f"{prefix}_largest_component_ratio": 0.0,
            f"{prefix}_solidity": 0.0,
            f"{prefix}_eccentricity": 0.0,
            f"{prefix}_perimeter_norm": 0.0,
            f"{prefix}_border_touch_fraction": 0.0,
        }
    lab = measure.label(b, connectivity=2)
    props = measure.regionprops(lab)
    largest = max(props, key=lambda p: p.area)
    border = np.zeros_like(b)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    boundary = morphology.dilation(b, morphology.disk(1)) ^ morphology.erosion(b, morphology.disk(1))
    return {
        f"{prefix}_fraction": float(b.sum() / total),
        f"{prefix}_components": float(lab.max()),
        f"{prefix}_largest_component_ratio": float(largest.area / max(1, b.sum())),
        f"{prefix}_solidity": float(largest.solidity),
        f"{prefix}_eccentricity": float(largest.eccentricity),
        f"{prefix}_perimeter_norm": float(largest.perimeter / max(1.0, math.sqrt(largest.area))),
        f"{prefix}_border_touch_fraction": float((boundary & border).sum() / max(1, boundary.sum())),
    }


def _boundary_strength(image: np.ndarray, region: np.ndarray, radius: int = 2) -> float:
    r = np.asarray(region, dtype=bool)
    if not r.any():
        return 0.0
    inner = r & ~morphology.erosion(r, morphology.disk(radius))
    outer = morphology.dilation(r, morphology.disk(radius)) & ~r
    if inner.sum() < 5 or outer.sum() < 5:
        return 0.0
    return float(abs(np.mean(image[inner]) - np.mean(image[outer])))


def _glcm_features(image01: np.ndarray, support: np.ndarray, levels: int = 32) -> dict[str, float]:
    rows, cols = np.where(support)
    if rows.size < 20:
        return {f"glcm_{name}": 0.0 for name in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "asm")}
    crop = image01[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()
    crop_support = support[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    fill = float(np.median(crop[crop_support])) if crop_support.any() else 0.0
    crop[~crop_support] = fill
    q = np.clip(np.floor(crop * (levels - 1)), 0, levels - 1).astype(np.uint8)
    glcm = feature.graycomatrix(
        q,
        distances=[1, 2],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True,
    )
    out: dict[str, float] = {}
    for name in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        out[f"glcm_{name.lower()}"] = float(np.nanmean(feature.graycoprops(glcm, name)))
    # Entropia de la matriz completa.
    p = glcm[glcm > 0]
    out["glcm_entropy"] = float(-(p * np.log2(p)).sum()) if p.size else 0.0
    return out


def _lbp_entropy(image01: np.ndarray, support: np.ndarray) -> float:
    q = np.round(image01 * 255).astype(np.uint8)
    lbp = feature.local_binary_pattern(q, P=8, R=1, method="uniform")
    return _entropy(lbp[support] / max(1.0, float(lbp.max())), bins=10)


def _cnr(image01: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    va, vb = _safe_values(image01, a), _safe_values(image01, b)
    denom = math.sqrt(float(va.var(ddof=1) + vb.var(ddof=1)) + 1e-12)
    return float(abs(va.mean() - vb.mean()) / denom)


def extract_features(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    include_morphology: bool = True,
    glcm_levels: int = 32,
) -> dict[str, float]:
    """Extrae indicadores interpretables de calidad de una imagen ecocardiografica.

    La imagen se normaliza de forma robusta solo para calcular indicadores comparables.
    La normalizacion no pretende mejorar visualmente la imagen. Las caracteristicas de
    textura deben recalcularse con parametros documentados en el experimento real.
    """

    x = robust_scale(np.asarray(image, dtype=np.float64))
    support = estimate_support(x)
    values = _safe_values(x, support)

    p01, p05, p10, p50, p90, p95, p99 = np.percentile(values, [1, 5, 10, 50, 90, 95, 99])
    lap = filters.laplace(x)
    gx = filters.sobel_h(x)
    gy = filters.sobel_v(x)
    grad2 = gx * gx + gy * gy
    edges = feature.canny(x, sigma=1.0)
    smooth = filters.gaussian(x, sigma=1.0, preserve_range=True)
    high = x - smooth
    high_values = _safe_values(high, support)
    noise_mad = float(np.median(np.abs(high_values - np.median(high_values))) / 0.6745)
    local_mean = ndi.uniform_filter(x, size=7)
    local_sq = ndi.uniform_filter(x * x, size=7)
    local_std = np.sqrt(np.maximum(local_sq - local_mean * local_mean, 0.0))

    features: dict[str, float] = {
        "intensity_mean": float(values.mean()),
        "intensity_std": float(values.std(ddof=1)),
        "intensity_median": float(p50),
        "intensity_p05": float(p05),
        "intensity_p95": float(p95),
        "dynamic_range_p95_p05": float(p95 - p05),
        "robust_range_p99_p01": float(p99 - p01),
        "intensity_skewness": float(skew(values, bias=False, nan_policy="omit")),
        "intensity_kurtosis": float(kurtosis(values, bias=False, nan_policy="omit")),
        "entropy_global": _entropy(values),
        "local_contrast_mean": float(np.mean(local_std[support])),
        "local_variance_mean": float(np.mean(local_std[support] ** 2)),
        "laplacian_variance": float(np.var(lap[support])),
        "tenengrad_mean": float(np.mean(grad2[support])),
        "edge_density": float(edges[support].mean()),
        "edge_strength_mean": float(np.mean(np.sqrt(grad2[support]))),
        "noise_mad_highpass": noise_mad,
        "high_frequency_energy": float(np.mean(high_values**2)),
        "speckle_index_global": float(values.std(ddof=1) / (abs(values.mean()) + 1e-8)),
        "dropout_fraction_global": float((values < max(0.04, p10 * 0.5)).mean()),
        "saturation_fraction": float(((values <= 0.01) | (values >= 0.99)).mean()),
        "support_fraction": float(support.mean()),
        "lbp_entropy": _lbp_entropy(x, support),
    }
    features.update(_glcm_features(x, support, levels=glcm_levels))

    if mask is not None:
        m = np.asarray(mask)
        if m.shape != x.shape:
            raise ValueError(f"La mascara {m.shape} no coincide con la imagen {x.shape}")
        lv = m == 1
        myo = m == 2
        la = m == 3
        if include_morphology:
            features.update(_binary_region_metrics(lv, "lv"))
            features.update(_binary_region_metrics(myo, "myo"))
            features.update(_binary_region_metrics(la, "la"))
            features["mask_foreground_fraction"] = float((m > 0).mean())
            features["myo_boundary_strength"] = _boundary_strength(x, myo)
            features["lv_boundary_strength"] = _boundary_strength(x, lv)
        if lv.any() and myo.any():
            features["cnr_myo_lv"] = _cnr(x, myo, lv)
            myo_values = _safe_values(x, myo)
            features["myo_speckle_index"] = float(myo_values.std(ddof=1) / (abs(myo_values.mean()) + 1e-8))
            features["dropout_fraction_myo"] = float((myo_values < max(0.05, np.percentile(myo_values, 10) * 0.55)).mean())
            features["myo_entropy"] = _entropy(myo_values)
        else:
            for name in ("cnr_myo_lv", "myo_speckle_index", "dropout_fraction_myo", "myo_entropy"):
                features[name] = 0.0
    # Limpieza defensiva para que los modelos no reciban infinito.
    for key, value in list(features.items()):
        if not np.isfinite(value):
            features[key] = 0.0
    return features


def feature_family(name: str) -> str:
    lower = name.lower()
    for family, prefixes in QUALITY_FEATURE_FAMILIES.items():
        if any(token in lower for token in prefixes):
            return family
    return "other"


def extract_feature_vector(
    image: np.ndarray,
    feature_names: Iterable[str],
    mask: np.ndarray | None = None,
) -> np.ndarray:
    values = extract_features(image, mask)
    return np.array([values.get(name, np.nan) for name in feature_names], dtype=np.float64)
