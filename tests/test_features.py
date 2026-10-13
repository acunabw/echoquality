from __future__ import annotations

import numpy as np

from echoquality.analysis.extractor import HierarchicalFeatureExtractor
from echoquality.data.synthetic import generate_echo_image


def test_hierarchical_extractor_and_identities() -> None:
    image, mask = generate_echo_image("Good", seed=12, size=96)
    extractor = HierarchicalFeatureExtractor()
    features = extractor.extract(image, mask).fused()
    assert any(name.startswith("global__") for name in features)
    assert any(name.startswith("anatomical__") for name in features)
    assert any(name.startswith("textural__") for name in features)
    assert np.isclose(
        features["global__robust_intensity_range_p95_p05"],
        features["global__intensity_p95"] - features["global__intensity_p05"],
    )
    assert np.isclose(
        features["anatomical__mask_foreground_fraction"],
        features["anatomical__lv_area_fraction"]
        + features["anatomical__myo_area_fraction"]
        + features["anatomical__la_area_fraction"],
    )
    assert np.isclose(
        features["textural__glcm_energy_sqrt_asm"] ** 2,
        features["textural__glcm_asm_uniformity"],
        atol=1e-10,
    )
    extractor.catalog().validate_values(features)
