from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import feature

from echoquality.analysis.feature_catalog import FeatureCatalog, definition
from echoquality.analysis.radiomics_matrices import (
    gldm_features,
    gldm_matrix,
    glcm_features,
    glrlm_features,
    glrlm_matrix,
    glszm_features,
    glszm_matrix,
    masked_glcm,
    ngtdm_features,
    quantize_roi,
)
from echoquality.analysis.support import safe_values


@dataclass(frozen=True)
class TextureFeatureConfig:
    levels: int = 32
    distances: tuple[int, ...] = (1, 2)
    angles_degrees: tuple[int, ...] = (0, 45, 90, 135)
    lbp_points: int = 8
    lbp_radius: float = 1.0
    lbp_method: str = "uniform"
    dropout_floor: float = 0.05
    enable_glrlm: bool = True
    enable_glszm: bool = True
    enable_gldm: bool = True
    enable_ngtdm: bool = True
    gldm_delta: int = 1
    gldm_alpha: int = 0
    ngtdm_delta: int = 1


def _entropy(values: np.ndarray, bins: int = 64) -> float:
    histogram, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    p = histogram.astype(float)
    if p.sum() == 0:
        return 0.0
    p /= p.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


class MyocardialTextureFeatures:
    """Level 3 representation calculated only inside the myocardial ROI."""

    def __init__(self, config: TextureFeatureConfig | None = None) -> None:
        self.config = config or TextureFeatureConfig()

    def extract(self, image01: np.ndarray, myocardium: np.ndarray) -> dict[str, float]:
        roi = np.asarray(myocardium, dtype=bool)
        if roi.shape != np.asarray(image01).shape:
            raise ValueError("Myocardial ROI and image shapes differ")
        values = safe_values(image01, roi)
        output: dict[str, float] = {
            "textural__myo_roi_fraction": float(roi.mean()),
            "textural__myo_entropy_first_order": _entropy(values),
            "textural__myo_speckle_index": float(values.std(ddof=1) / max(abs(values.mean()), 1e-8)) if values.size > 1 else 0.0,
            "textural__myo_dropout_fraction": float((values < max(self.config.dropout_floor, np.percentile(values, 10) * 0.55)).mean()),
        }
        if roi.sum() < 20:
            return output
        quantized = quantize_roi(image01, roi, self.config.levels)
        angles = tuple(np.deg2rad(value) for value in self.config.angles_degrees)
        glcm = masked_glcm(
            quantized,
            roi,
            levels=self.config.levels,
            distances=self.config.distances,
            angles=angles,
            symmetric=True,
        )
        for name, value in glcm_features(glcm).items():
            output[f"textural__glcm_{name}"] = value

        uint8 = np.round(np.clip(image01, 0.0, 1.0) * 255).astype(np.uint8)
        lbp = feature.local_binary_pattern(
            uint8,
            P=self.config.lbp_points,
            R=self.config.lbp_radius,
            method=self.config.lbp_method,
        )
        lbp_values = lbp[roi]
        bins = self.config.lbp_points + 2 if self.config.lbp_method == "uniform" else 2 ** self.config.lbp_points
        histogram, _ = np.histogram(lbp_values, bins=bins, range=(0, bins))
        probability = histogram.astype(float) / max(1.0, float(histogram.sum()))
        probability = probability[probability > 0]
        output["textural__lbp_entropy"] = float(-(probability * np.log2(probability)).sum()) if probability.size else 0.0

        n_pixels = int(roi.sum())
        if self.config.enable_glrlm:
            matrix = glrlm_matrix(quantized, roi, levels=self.config.levels)
            for name, value in glrlm_features(matrix, n_pixels).items():
                output[f"textural__glrlm_{name}"] = value
        if self.config.enable_glszm:
            matrix = glszm_matrix(quantized, roi, levels=self.config.levels)
            for name, value in glszm_features(matrix, n_pixels).items():
                output[f"textural__glszm_{name}"] = value
        if self.config.enable_gldm:
            matrix = gldm_matrix(
                quantized,
                roi,
                levels=self.config.levels,
                delta=self.config.gldm_delta,
                alpha=self.config.gldm_alpha,
            )
            for name, value in gldm_features(matrix, n_pixels).items():
                output[f"textural__gldm_{name}"] = value
        if self.config.enable_ngtdm:
            for name, value in ngtdm_features(
                quantized,
                roi,
                levels=self.config.levels,
                delta=self.config.ngtdm_delta,
            ).items():
                output[f"textural__ngtdm_{name}"] = value
        return {key: (float(value) if np.isfinite(value) else 0.0) for key, value in output.items()}

    def definitions(self) -> FeatureCatalog:
        support = "myocardial ROI"
        defs = [
            definition("textural__myo_roi_fraction", "textural", "roi_audit", "MYO pixels/image pixels", "Texture ROI coverage.", support, source_type="custom", source="EchoQuality audit"),
            definition("textural__myo_entropy_first_order", "textural", "first_order", "-sum p(k)log2 p(k)", "Myocardial intensity entropy; noise can increase it.", support, units="bits", source_type="direct_GW", source="Gonzalez & Woods 4e, Eq. 11-31", aliases=("myo_entropy",)),
            definition("textural__myo_speckle_index", "textural", "ultrasound_texture", "std(MYO)/mean(MYO)", "Relative myocardial heterogeneity/speckle.", support, source_type="external", source="Ultrasound-specific/custom", aliases=("myo_speckle_index",)),
            definition("textural__myo_dropout_fraction", "textural", "signal_loss", "fraction(MYO intensity < adaptive threshold)", "Low-signal myocardial fraction.", support, source_type="external", source="Ultrasound-specific/custom", aliases=("dropout_fraction_myo",)),
            definition("textural__glcm_contrast", "textural", "GLCM", "sum (i-j)^2 P(i,j)", "Local pairwise intensity differences.", support, source_type="direct_GW", source="G&W Table 11.3"),
            definition("textural__glcm_dissimilarity", "textural", "GLCM", "sum |i-j| P(i,j)", "Linear pairwise intensity difference.", support, source_type="external_custom", source="Common GLCM extension; not explicit in G&W Table 11.3"),
            definition("textural__glcm_homogeneity", "textural", "GLCM", "sum P(i,j)/(1+(i-j)^2)", "Concentration near the GLCM diagonal.", support, source_type="direct_concept_GW", source="G&W Table 11.3; implementation convention documented"),
            definition("textural__glcm_correlation", "textural", "GLCM", "sum[(i-mu_i)(j-mu_j)P]/(sigma_i sigma_j)", "Linear dependence of neighboring gray levels.", support, source_type="direct_GW", source="G&W Table 11.3"),
            definition("textural__glcm_asm_uniformity", "textural", "GLCM", "sum P(i,j)^2", "Angular second moment/uniformity. This is the quantity G&W also calls energy.", support, source_type="direct_GW", source="G&W Table 11.3", aliases=("glcm_asm",)),
            definition("textural__glcm_energy_sqrt_asm", "textural", "GLCM", "sqrt(sum P(i,j)^2)", "scikit-image-style Energy; square equals ASM within numerical tolerance.", support, source_type="library_convention", source="scikit-image convention; terminology differs from G&W", aliases=("glcm_energy",)),
            definition("textural__glcm_entropy", "textural", "GLCM", "-sum P(i,j)log2P(i,j)", "Spatial co-occurrence uncertainty.", support, units="bits", source_type="direct_GW", source="G&W Table 11.3"),
            definition("textural__lbp_entropy", "textural", "LBP", "entropy of LBP-code histogram", "Diversity of local binary patterns.", support, source_type="external", source="Ojala et al. (2002)"),
        ]
        for family, names in {
            "GLRLM": ("sre", "lre", "gln", "rln", "rp", "lgre", "hgre"),
            "GLSZM": ("sze", "lze", "gln", "zsn", "zp", "lgze", "hgze"),
            "GLDM": ("sde", "lde", "gln", "dn", "dp", "lgle", "hgle"),
            "NGTDM": ("coarseness", "contrast", "busyness", "complexity", "strength"),
        }.items():
            prefix = family.lower()
            for name in names:
                defs.append(
                    definition(
                        f"textural__{prefix}_{name}",
                        "textural",
                        family,
                        f"standard {family} {name.upper()} formula implemented in radiomics_matrices.py",
                        f"{family} descriptor {name.upper()} within myocardium.",
                        support,
                        source_type="external_radiomics",
                        source="Radiomics-style matrix definition; exact implementation shipped with EchoQuality 2.0",
                    )
                )
        return FeatureCatalog(defs)
