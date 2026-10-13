from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import measure, morphology

from echoquality.analysis.feature_catalog import FeatureCatalog, definition
from echoquality.analysis.roi import AnatomicalRegions
from echoquality.analysis.support import safe_values


@dataclass(frozen=True)
class MorphologyFeatureConfig:
    connectivity: int = 2
    entropy_bins: int = 64


def _entropy(values: np.ndarray, bins: int) -> float:
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    probability = hist.astype(float)
    if probability.sum() == 0:
        return 0.0
    probability /= probability.sum()
    probability = probability[probability > 0]
    return float(-(probability * np.log2(probability)).sum())


def _thickness(binary: np.ndarray) -> tuple[float, float, float]:
    mask = np.asarray(binary, dtype=bool)
    if not mask.any():
        return 0.0, 0.0, 0.0
    distance = ndi.distance_transform_edt(mask)
    skeleton = morphology.skeletonize(mask)
    values = 2.0 * distance[skeleton]
    if values.size == 0:
        values = 2.0 * distance[mask]
    return float(np.mean(values)), float(np.median(values)), float(np.max(values))


def _region_shape(binary: np.ndarray) -> dict[str, float]:
    mask = np.asarray(binary, dtype=bool)
    total = float(mask.size)
    if not mask.any():
        return {
            "area_px": 0.0,
            "area_fraction": 0.0,
            "perimeter_px": 0.0,
            "circularity": 0.0,
            "compactness": 0.0,
            "aspect_ratio": 0.0,
            "eccentricity": 0.0,
            "solidity": 0.0,
            "components": 0.0,
            "largest_component_ratio": 0.0,
            "border_touch_fraction": 0.0,
            "thickness_mean_px": 0.0,
            "thickness_median_px": 0.0,
            "thickness_max_px": 0.0,
        }
    labels = measure.label(mask, connectivity=2)
    properties = measure.regionprops(labels)
    largest = max(properties, key=lambda region: region.area)
    largest_mask = labels == largest.label
    area = float(largest.area)
    perimeter = float(measure.perimeter_crofton(largest_mask, directions=4))
    circularity = float(4.0 * math.pi * area / max(perimeter * perimeter, 1e-12))
    compactness = float(perimeter * perimeter / max(4.0 * math.pi * area, 1e-12))
    minor = float(largest.axis_minor_length)
    aspect_ratio = float(largest.axis_major_length / max(minor, 1e-12))
    boundary = morphology.dilation(largest_mask, morphology.disk(1)) ^ morphology.erosion(largest_mask, morphology.disk(1))
    image_border = np.zeros_like(mask, dtype=bool)
    image_border[[0, -1], :] = True
    image_border[:, [0, -1]] = True
    thickness_mean, thickness_median, thickness_max = _thickness(largest_mask)
    return {
        "area_px": area,
        "area_fraction": float(mask.sum() / total),
        "perimeter_px": perimeter,
        "circularity": circularity,
        "compactness": compactness,
        "aspect_ratio": aspect_ratio,
        "eccentricity": float(largest.eccentricity),
        "solidity": float(largest.solidity),
        "components": float(labels.max()),
        "largest_component_ratio": float(area / max(1.0, float(mask.sum()))),
        "border_touch_fraction": float((boundary & image_border).sum() / max(1, boundary.sum())),
        "thickness_mean_px": thickness_mean,
        "thickness_median_px": thickness_median,
        "thickness_max_px": thickness_max,
    }


def _region_intensity(image01: np.ndarray, binary: np.ndarray, bins: int) -> dict[str, float]:
    values = safe_values(image01, binary)
    p05, p50, p95 = np.percentile(values, [5, 50, 95])
    return {
        "intensity_mean": float(values.mean()),
        "intensity_median": float(p50),
        "intensity_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "intensity_p05": float(p05),
        "intensity_p95": float(p95),
        "intensity_entropy": _entropy(values, bins),
    }


class AnatomicalQualityFeatures:
    """Level 2 representation: mask geometry and region-conditioned intensity."""

    def __init__(self, config: MorphologyFeatureConfig | None = None) -> None:
        self.config = config or MorphologyFeatureConfig()

    def extract(
        self,
        image01: np.ndarray,
        regions: AnatomicalRegions,
        *,
        spacing_yx: tuple[float, float] | None = None,
    ) -> dict[str, float]:
        output: dict[str, float] = {}
        region_map = {"lv": regions.lv, "myo": regions.myocardium, "la": regions.la}
        for name, binary in region_map.items():
            for key, value in _region_shape(binary).items():
                output[f"anatomical__{name}_{key}"] = value
            for key, value in _region_intensity(image01, binary, self.config.entropy_bins).items():
                output[f"anatomical__{name}_{key}"] = value
            if spacing_yx is not None:
                spacing_y, spacing_x = spacing_yx
                output[f"anatomical__{name}_area_mm2"] = output[f"anatomical__{name}_area_px"] * spacing_y * spacing_x
                output[f"anatomical__{name}_perimeter_mm_approx"] = output[f"anatomical__{name}_perimeter_px"] * 0.5 * (spacing_y + spacing_x)
                output[f"anatomical__{name}_thickness_mean_mm_approx"] = output[f"anatomical__{name}_thickness_mean_px"] * 0.5 * (spacing_y + spacing_x)
        output["anatomical__mask_foreground_fraction"] = float(regions.foreground.mean())
        output["anatomical__fraction_sum_identity"] = float(
            output["anatomical__lv_area_fraction"]
            + output["anatomical__myo_area_fraction"]
            + output["anatomical__la_area_fraction"]
        )
        return {key: (float(value) if np.isfinite(value) else 0.0) for key, value in output.items()}

    def definitions(self, *, include_physical: bool = True) -> FeatureCatalog:
        defs = []
        gonzalez_shape = "Gonzalez & Woods 4e, Ch. 9 and Sec. 11.4"
        for region in ("lv", "myo", "la"):
            prefix = f"anatomical__{region}_"
            support = f"{region.upper()} mask"
            defs.extend(
                [
                    definition(prefix + "area_px", "anatomical", "shape", "number of pixels in largest component", "Region area in pixels.", support, units="pixel^2", source_type="direct_GW", source=gonzalez_shape),
                    definition(prefix + "area_fraction", "anatomical", "shape", "region pixels / image pixels", "Normalized region coverage.", support, source_type="derived_GW", source=gonzalez_shape),
                    definition(prefix + "perimeter_px", "anatomical", "shape", "Crofton perimeter", "Boundary length in pixels.", support, units="pixel", source_type="direct_GW", source=gonzalez_shape),
                    definition(prefix + "circularity", "anatomical", "shape", "4*pi*A/P^2", "One for an ideal circle; smaller for irregular or elongated shapes.", support, source_type="derived_GW", source=gonzalez_shape),
                    definition(prefix + "compactness", "anatomical", "shape", "P^2/(4*pi*A)", "Inverse circularity; exact convention is documented.", support, source_type="derived_GW", source=gonzalez_shape),
                    definition(prefix + "aspect_ratio", "anatomical", "shape", "major_axis/minor_axis", "Elongation of the equivalent ellipse.", support, source_type="derived_GW", source=gonzalez_shape),
                    definition(prefix + "eccentricity", "anatomical", "shape", "eccentricity of equal-second-moment ellipse", "Ellipse eccentricity.", support, source_type="direct_GW", source=gonzalez_shape),
                    definition(prefix + "solidity", "anatomical", "shape", "A/A_convex_hull", "Convex-hull occupancy; low MYO solidity may be physiological.", support, source_type="external_custom", source="Convex hull from G&W; solidity ratio custom", confounders=("normal myocardial ring geometry",)),
                    definition(prefix + "components", "anatomical", "topology", "number of 8-connected components", "Mask fragmentation count.", support, source_type="direct_GW", source=gonzalez_shape, parameters={"connectivity": 8}),
                    definition(prefix + "largest_component_ratio", "anatomical", "topology", "largest component area / total mask area", "Dominance of the primary component.", support, source_type="derived_GW", source=gonzalez_shape),
                    definition(prefix + "border_touch_fraction", "anatomical", "coverage", "boundary pixels touching image border / boundary pixels", "Potential truncation or incomplete field of view.", support, source_type="custom", source="Boundary concepts from G&W; ratio custom"),
                    definition(prefix + "thickness_mean_px", "anatomical", "thickness", "mean(2*distance transform on skeleton)", "Approximate local thickness.", support, units="pixel", source_type="custom", source="Distance-transform/skeleton implementation"),
                    definition(prefix + "thickness_median_px", "anatomical", "thickness", "median(2*distance transform on skeleton)", "Robust approximate thickness.", support, units="pixel", source_type="custom", source="Distance-transform/skeleton implementation"),
                    definition(prefix + "thickness_max_px", "anatomical", "thickness", "max(2*distance transform on skeleton)", "Maximum approximate thickness.", support, units="pixel", source_type="custom", source="Distance-transform/skeleton implementation"),
                    definition(prefix + "intensity_mean", "anatomical", "roi_intensity", "mean(I in ROI)", "Region-conditioned mean intensity.", support, source_type="direct_GW", source="G&W histogram statistics"),
                    definition(prefix + "intensity_median", "anatomical", "roi_intensity", "p50(I in ROI)", "Region-conditioned median intensity.", support, source_type="derived_GW", source="G&W histogram/CDF"),
                    definition(prefix + "intensity_std", "anatomical", "roi_intensity", "std(I in ROI)", "Region-conditioned intensity dispersion.", support, source_type="direct_GW", source="G&W histogram statistics"),
                    definition(prefix + "intensity_p05", "anatomical", "roi_intensity", "p05(I in ROI)", "Region dark-tail percentile.", support, source_type="derived_GW", source="G&W histogram/CDF"),
                    definition(prefix + "intensity_p95", "anatomical", "roi_intensity", "p95(I in ROI)", "Region bright-tail percentile.", support, source_type="derived_GW", source="G&W histogram/CDF"),
                    definition(prefix + "intensity_entropy", "anatomical", "roi_intensity", "-sum p log2 p in ROI", "Region intensity entropy, not a standalone quality measure.", support, units="bits", source_type="direct_GW", source="G&W Eq. 11-31"),
                ]
            )
            if include_physical:
                defs.extend(
                    [
                        definition(prefix + "area_mm2", "anatomical", "physical_geometry", "area_px*spacing_x*spacing_y", "Approximate physical region area when spacing exists.", support, units="mm^2", source_type="derived_GW", source="Area plus image spacing metadata"),
                        definition(prefix + "perimeter_mm_approx", "anatomical", "physical_geometry", "perimeter_px*mean spacing", "Approximate physical perimeter; exact only for near-isotropic spacing.", support, units="mm", source_type="custom", source="Approximation documented by implementation"),
                        definition(prefix + "thickness_mean_mm_approx", "anatomical", "physical_geometry", "thickness_px*mean spacing", "Approximate physical thickness.", support, units="mm", source_type="custom", source="Approximation documented by implementation"),
                    ]
                )
        defs.extend(
            [
                definition("anatomical__mask_foreground_fraction", "anatomical", "coverage", "foreground pixels / image pixels", "Total anatomical mask coverage.", "LV+MYO+LA masks", source_type="derived_GW", source=gonzalez_shape, aliases=("mask_foreground_fraction",)),
                definition("anatomical__fraction_sum_identity", "anatomical", "audit", "LV fraction + MYO fraction + LA fraction", "Auditable identity for non-overlapping labels.", "LV+MYO+LA masks", source_type="custom", source="EchoQuality audit feature"),
            ]
        )
        return FeatureCatalog(defs)
