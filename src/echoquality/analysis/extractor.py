from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from echoquality.analysis.border_features import BorderFeatureConfig, BorderQualityFeatures
from echoquality.analysis.feature_catalog import FeatureCatalog, definition
from echoquality.analysis.global_features import GlobalFeatureConfig, GlobalQualityFeatures
from echoquality.analysis.morphology_features import AnatomicalQualityFeatures, MorphologyFeatureConfig
from echoquality.analysis.roi import regions_from_camus_mask
from echoquality.analysis.support import estimate_ultrasound_support
from echoquality.analysis.texture_features import MyocardialTextureFeatures, TextureFeatureConfig
from echoquality.data.io import robust_scale
from echoquality.types import HierarchicalFeatures, ImageMetadata


def _dataclass_from_mapping(cls: type, mapping: dict[str, Any] | None):
    mapping = mapping or {}
    allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
    return cls(**{key: value for key, value in mapping.items() if key in allowed})


class HierarchicalFeatureExtractor:
    """Extract independent global, anatomical, and myocardial texture patterns."""

    def __init__(
        self,
        *,
        global_config: GlobalFeatureConfig | None = None,
        morphology_config: MorphologyFeatureConfig | None = None,
        border_config: BorderFeatureConfig | None = None,
        texture_config: TextureFeatureConfig | None = None,
    ) -> None:
        self.global_extractor = GlobalQualityFeatures(global_config)
        self.anatomical_extractor = AnatomicalQualityFeatures(morphology_config)
        self.border_extractor = BorderQualityFeatures(border_config)
        self.texture_extractor = MyocardialTextureFeatures(texture_config)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "HierarchicalFeatureExtractor":
        feature_config = config.get("features", config)
        return cls(
            global_config=_dataclass_from_mapping(GlobalFeatureConfig, feature_config.get("global")),
            morphology_config=_dataclass_from_mapping(MorphologyFeatureConfig, feature_config.get("anatomical")),
            border_config=_dataclass_from_mapping(BorderFeatureConfig, feature_config.get("border")),
            texture_config=_dataclass_from_mapping(TextureFeatureConfig, feature_config.get("textural")),
        )

    def extract(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        *,
        metadata: ImageMetadata | None = None,
    ) -> HierarchicalFeatures:
        raw = np.asarray(image, dtype=np.float64)
        support = estimate_ultrasound_support(raw)
        normalized = robust_scale(raw, support=support)
        global_features = self.global_extractor.extract(raw, support=support)
        anatomical: dict[str, float] = {"anatomical__mask_available": float(mask is not None)}
        textural: dict[str, float] = {"textural__myo_roi_available": 0.0}
        if mask is not None:
            regions = regions_from_camus_mask(mask, shape=raw.shape)
            spacing = metadata.spacing_yx if metadata else None
            anatomical.update(self.anatomical_extractor.extract(normalized, regions, spacing_yx=spacing))
            anatomical.update(self.border_extractor.extract(normalized, regions))
            textural["textural__myo_roi_available"] = float(regions.myocardium.any())
            textural.update(self.texture_extractor.extract(normalized, regions.myocardium))
        return HierarchicalFeatures(global_features, anatomical, textural)

    def catalog(self) -> FeatureCatalog:
        catalog = FeatureCatalog()
        catalog.extend(self.global_extractor.definitions().definitions())
        catalog.extend(self.anatomical_extractor.definitions().definitions())
        catalog.extend(self.border_extractor.definitions().definitions())
        catalog.extend(self.texture_extractor.definitions().definitions())
        catalog.extend(
            [
                definition("anatomical__mask_available", "anatomical", "availability", "1 if mask is available", "Availability flag for anatomical representation.", "metadata", source_type="custom"),
                definition("textural__myo_roi_available", "textural", "availability", "1 if MYO ROI has pixels", "Availability flag for myocardial texture representation.", "metadata", source_type="custom"),
            ]
        )
        return catalog
