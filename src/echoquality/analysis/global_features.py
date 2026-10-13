from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import kurtosis, skew
from skimage import feature, filters

from echoquality.analysis.feature_catalog import FeatureCatalog, definition
from echoquality.analysis.support import estimate_ultrasound_support, safe_values
from echoquality.data.io import robust_scale


@dataclass(frozen=True)
class GlobalFeatureConfig:
    histogram_bins: int = 64
    local_window: int = 7
    edge_sigma: float = 1.0
    highpass_sigma: float = 1.0
    dropout_floor: float = 0.04
    saturation_low: float = 0.01
    saturation_high: float = 0.99


def _entropy(values: np.ndarray, bins: int) -> float:
    histogram, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    probabilities = histogram.astype(float)
    if probabilities.sum() == 0:
        return 0.0
    probabilities /= probabilities.sum()
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


class GlobalQualityFeatures:
    """Level 1 representation: acquisition-wide intensity and signal quality."""

    def __init__(self, config: GlobalFeatureConfig | None = None) -> None:
        self.config = config or GlobalFeatureConfig()

    def extract(self, image: np.ndarray, *, support: np.ndarray | None = None) -> dict[str, float]:
        raw = np.asarray(image, dtype=np.float64)
        support_mask = estimate_ultrasound_support(raw) if support is None else np.asarray(support, dtype=bool)
        normalized = robust_scale(raw, support=support_mask)
        values = safe_values(normalized, support_mask)
        p01, p05, p10, p50, p90, p95, p99 = np.percentile(values, [1, 5, 10, 50, 90, 95, 99])

        laplacian = filters.laplace(normalized)
        sobel_h = filters.sobel_h(normalized)
        sobel_v = filters.sobel_v(normalized)
        gradient_squared = sobel_h * sobel_h + sobel_v * sobel_v
        gradient = np.sqrt(gradient_squared)
        edges = feature.canny(normalized, sigma=self.config.edge_sigma)
        smooth = filters.gaussian(normalized, sigma=self.config.highpass_sigma, preserve_range=True)
        highpass = normalized - smooth
        high_values = safe_values(highpass, support_mask)
        noise_mad = float(1.4826 * np.median(np.abs(high_values - np.median(high_values))))

        window = max(3, int(self.config.local_window) | 1)
        local_mean = ndi.uniform_filter(normalized, size=window)
        local_square = ndi.uniform_filter(normalized * normalized, size=window)
        local_variance = np.maximum(local_square - local_mean * local_mean, 0.0)
        local_std = np.sqrt(local_variance)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        high_frequency_energy = float(np.mean(high_values**2))
        total_energy = float(np.mean(values**2))

        output = {
            "global__brightness_mean": mean,
            "global__intensity_std": std,
            "global__intensity_median": float(p50),
            "global__intensity_p05": float(p05),
            "global__intensity_p95": float(p95),
            "global__robust_intensity_range_p95_p05": float(p95 - p05),
            "global__robust_intensity_range_p99_p01": float(p99 - p01),
            "global__intensity_skewness": float(skew(values, bias=False, nan_policy="omit")) if values.size > 2 else 0.0,
            "global__intensity_kurtosis_excess": float(kurtosis(values, bias=False, nan_policy="omit")) if values.size > 3 else 0.0,
            "global__histogram_entropy": _entropy(values, self.config.histogram_bins),
            "global__local_contrast_std_mean": float(np.mean(local_std[support_mask])),
            "global__local_variance_mean": float(np.mean(local_variance[support_mask])),
            "global__laplacian_variance": float(np.var(laplacian[support_mask])),
            "global__tenengrad_mean": float(np.mean(gradient_squared[support_mask])),
            "global__edge_density": float(edges[support_mask].mean()),
            "global__edge_strength_mean": float(np.mean(gradient[support_mask])),
            "global__noise_mad_highpass": noise_mad,
            "global__high_frequency_energy": high_frequency_energy,
            "global__high_frequency_energy_ratio": float(high_frequency_energy / max(total_energy, 1e-12)),
            "global__speckle_index": float(std / max(abs(mean), 1e-8)),
            "global__dropout_fraction": float((values < max(self.config.dropout_floor, p10 * 0.5)).mean()),
            "global__saturation_fraction": float(
                ((values <= self.config.saturation_low) | (values >= self.config.saturation_high)).mean()
            ),
            "global__support_fraction": float(support_mask.mean()),
        }
        return {key: (float(value) if np.isfinite(value) else 0.0) for key, value in output.items()}

    def definitions(self) -> FeatureCatalog:
        gonzalez_hist = "Gonzalez & Woods 4e, Sec. 3.3 and Sec. 11.4"
        gonzalez_edges = "Gonzalez & Woods 4e, Sec. 3.6 and Ch. 10"
        defs = [
            definition("global__brightness_mean", "global", "intensity", "mu = mean(I in support)", "Average normalized brightness.", "ultrasound support", source_type="direct_GW", source=gonzalez_hist, confounders=("gain", "depth", "normalization")),
            definition("global__intensity_std", "global", "intensity", "sigma = std(I in support)", "Global dispersion; contrast and noise both contribute.", "ultrasound support", source_type="direct_GW", source=gonzalez_hist, confounders=("speckle", "gain")),
            definition("global__intensity_median", "global", "intensity", "p50", "Robust center of the intensity distribution.", "ultrasound support", source_type="derived_GW", source=gonzalez_hist),
            definition("global__intensity_p05", "global", "intensity", "p05", "Robust dark-tail percentile.", "ultrasound support", source_type="derived_GW", source=gonzalez_hist),
            definition("global__intensity_p95", "global", "intensity", "p95", "Robust bright-tail percentile.", "ultrasound support", source_type="derived_GW", source=gonzalez_hist),
            definition("global__robust_intensity_range_p95_p05", "global", "intensity", "p95 - p05", "Robust percentile intensity span; not physical system dynamic range.", "ultrasound support", source_type="custom", source="Derived from histogram percentiles; distinct from G&W physical dynamic range", aliases=("dynamic_range_p95_p05",)),
            definition("global__robust_intensity_range_p99_p01", "global", "intensity", "p99 - p01", "Wide robust percentile intensity span.", "ultrasound support", source_type="custom", source="Derived from histogram percentiles", aliases=("robust_range_p99_p01",)),
            definition("global__intensity_skewness", "global", "intensity", "E[(I-mu)^3]/sigma^3", "Distribution asymmetry.", "ultrasound support", source_type="direct_concept_GW", source=gonzalez_hist),
            definition("global__intensity_kurtosis_excess", "global", "intensity", "E[(I-mu)^4]/sigma^4 - 3", "Tail/peak shape using the excess-kurtosis convention.", "ultrasound support", source_type="direct_concept_GW", source=gonzalez_hist),
            definition("global__histogram_entropy", "global", "intensity", "-sum p(k) log2 p(k)", "Intensity uncertainty; not a standalone quality score.", "ultrasound support", units="bits", source_type="direct_GW", source="Gonzalez & Woods 4e, Eq. 11-31", confounders=("noise", "quantization", "CLAHE")),
            definition("global__local_contrast_std_mean", "global", "local_statistics", "mean(local standard deviation)", "Average local contrast in a fixed window.", "ultrasound support", source_type="derived_GW", source="Gonzalez & Woods 4e, local statistics Sec. 3.3", parameters={"window": self.config.local_window}),
            definition("global__local_variance_mean", "global", "local_statistics", "mean(local variance)", "Local heterogeneity from signal or noise.", "ultrasound support", source_type="direct_GW", source="Gonzalez & Woods 4e, local variance Eq. 3-28", parameters={"window": self.config.local_window}),
            definition("global__laplacian_variance", "global", "sharpness", "Var(Laplacian(I))", "Sharpness/detail proxy with high noise sensitivity.", "ultrasound support", source_type="derived_GW", source=gonzalez_edges),
            definition("global__tenengrad_mean", "global", "sharpness", "mean(Gx^2 + Gy^2), Sobel", "Gradient-energy focus proxy.", "ultrasound support", source_type="external", source="Tenengrad; Sobel basis in G&W", confounders=("speckle", "noise")),
            definition("global__edge_density", "global", "edges", "edge pixels / support pixels", "Amount of detected edge structure.", "ultrasound support", source_type="derived_GW", source=gonzalez_edges, parameters={"canny_sigma": self.config.edge_sigma}),
            definition("global__edge_strength_mean", "global", "edges", "mean(|gradient I|)", "Mean gradient strength.", "ultrasound support", source_type="derived_GW", source=gonzalez_edges),
            definition("global__noise_mad_highpass", "global", "noise", "1.4826 * MAD(I - Gaussian(I))", "Robust high-frequency residual scale.", "ultrasound support", source_type="custom", source="High-pass concept from G&W; MAD estimator custom", parameters={"sigma": self.config.highpass_sigma}),
            definition("global__high_frequency_energy", "global", "frequency", "mean((I - Gaussian(I))^2)", "Fine-scale energy from edges, texture, and noise.", "ultrasound support", source_type="derived_GW", source="Gonzalez & Woods 4e, Ch. 4", parameters={"sigma": self.config.highpass_sigma}),
            definition("global__high_frequency_energy_ratio", "global", "frequency", "high-frequency energy / total energy", "Relative fine-scale energy.", "ultrasound support", source_type="custom", source="Derived from G&W frequency concepts"),
            definition("global__speckle_index", "global", "noise", "std(I)/mean(I)", "Global coefficient of variation; mixes tissue heterogeneity and speckle.", "ultrasound support", source_type="external", source="Ultrasound-specific/custom", aliases=("speckle_index_global",)),
            definition("global__dropout_fraction", "global", "signal_loss", "fraction(I < adaptive threshold)", "Low-signal fraction inside support.", "ultrasound support", source_type="external", source="Ultrasound-specific/custom", parameters={"floor": self.config.dropout_floor}, aliases=("dropout_fraction_global",)),
            definition("global__saturation_fraction", "global", "intensity", "fraction(I <= low or I >= high)", "Normalized clipping fraction.", "ultrasound support", source_type="derived_GW", source="G&W saturation concept; threshold custom", parameters={"low": self.config.saturation_low, "high": self.config.saturation_high}),
            definition("global__support_fraction", "global", "support", "support pixels / image pixels", "Acquisition-field coverage, not anatomical coverage.", "full image", source_type="custom", source="EchoQuality support heuristic"),
        ]
        return FeatureCatalog(defs)
