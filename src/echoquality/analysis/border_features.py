from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from skimage import filters, morphology

from echoquality.analysis.feature_catalog import FeatureCatalog, definition
from echoquality.analysis.roi import AnatomicalRegions
from echoquality.analysis.support import safe_values


@dataclass(frozen=True)
class BorderFeatureConfig:
    boundary_radius: int = 2
    continuity_percentile: float = 50.0


def _boundary_bands(region: np.ndarray, radius: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.asarray(region, dtype=bool)
    footprint = morphology.disk(radius)
    inner = mask & ~morphology.erosion(mask, footprint)
    outer = morphology.dilation(mask, footprint) & ~mask
    boundary = morphology.dilation(mask, morphology.disk(1)) ^ morphology.erosion(mask, morphology.disk(1))
    return inner, outer, boundary


class BorderQualityFeatures:
    """Anatomically conditioned boundary visibility and region separation."""

    def __init__(self, config: BorderFeatureConfig | None = None) -> None:
        self.config = config or BorderFeatureConfig()

    def extract(self, image01: np.ndarray, regions: AnatomicalRegions) -> dict[str, float]:
        gradient = filters.sobel(np.asarray(image01, dtype=float))
        output: dict[str, float] = {}
        region_map = {"lv": regions.lv, "myo": regions.myocardium, "la": regions.la}
        global_threshold = float(np.percentile(gradient[np.isfinite(gradient)], self.config.continuity_percentile))
        for name, region in region_map.items():
            inner, outer, boundary = _boundary_bands(region, self.config.boundary_radius)
            if inner.sum() < 5 or outer.sum() < 5 or boundary.sum() < 5:
                contrast = strength = continuity = 0.0
            else:
                contrast = float(abs(np.mean(image01[inner]) - np.mean(image01[outer])))
                strength = float(np.mean(gradient[boundary]))
                continuity = float((gradient[boundary] >= global_threshold).mean())
            output[f"anatomical__{name}_boundary_contrast"] = contrast
            output[f"anatomical__{name}_boundary_gradient_mean"] = strength
            output[f"anatomical__{name}_boundary_continuity"] = continuity
        myo_values = safe_values(image01, regions.myocardium)
        lv_values = safe_values(image01, regions.lv)
        variance_sum = float(myo_values.var(ddof=1) + lv_values.var(ddof=1)) if myo_values.size > 1 and lv_values.size > 1 else 0.0
        output["anatomical__cnr_myo_lv"] = float(abs(myo_values.mean() - lv_values.mean()) / math.sqrt(0.5 * variance_sum + 1e-12))
        return {key: (float(value) if np.isfinite(value) else 0.0) for key, value in output.items()}

    def definitions(self) -> FeatureCatalog:
        defs = []
        for region in ("lv", "myo", "la"):
            support = f"{region.upper()} boundary"
            defs.extend(
                [
                    definition(f"anatomical__{region}_boundary_contrast", "anatomical", "boundary", "|mean(inner band)-mean(outer band)|", "Intensity separation across the anatomical boundary.", support, source_type="custom_derived_GW", source="G&W boundary and intensity statistics", parameters={"radius": self.config.boundary_radius}),
                    definition(f"anatomical__{region}_boundary_gradient_mean", "anatomical", "boundary", "mean(Sobel magnitude on boundary)", "Boundary visibility in the image.", support, source_type="custom_derived_GW", source="G&W Sobel/gradient and boundary concepts"),
                    definition(f"anatomical__{region}_boundary_continuity", "anatomical", "boundary", "fraction(boundary gradient >= global percentile)", "Continuity of visible boundary evidence.", support, source_type="custom", source="EchoQuality 2.0", parameters={"percentile": self.config.continuity_percentile}),
                ]
            )
        defs.append(
            definition("anatomical__cnr_myo_lv", "anatomical", "region_separation", "|mu_myo-mu_lv|/sqrt((var_myo+var_lv)/2)", "Intensity separability between myocardium and LV cavity.", "MYO and LV masks", source_type="custom", source="CNR formula documented by EchoQuality; G&W supplies contrast/SNR concepts", aliases=("cnr_myo_lv",))
        )
        return FeatureCatalog(defs)
