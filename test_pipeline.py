from __future__ import annotations

import numpy as np
import pandas as pd

from echo_quality_pipeline.associations import association_table, select_nonredundant
from echo_quality_pipeline.camus import normalize_quality
from echo_quality_pipeline.features import extract_features
from echo_quality_pipeline.optimization import optimize_image
from echo_quality_pipeline.reference import GoodReference
from echo_quality_pipeline.synthetic import generate_echo_image
from echo_quality_pipeline.transforms import TransformationSpec, apply_transformation


def test_quality_aliases() -> None:
    assert normalize_quality("Good") == "Good"
    assert normalize_quality("media") == "Medium"
    assert normalize_quality("bad") == "Poor"


def test_feature_extraction_is_finite() -> None:
    image, mask = generate_echo_image("Good", seed=42)
    features = extract_features(image, mask)
    assert len(features) >= 45
    assert all(np.isfinite(value) for value in features.values())
    assert features["cnr_myo_lv"] > 0


def test_association_and_nonredundancy() -> None:
    rows = []
    for i, quality in enumerate(["Poor"] * 10 + ["Medium"] * 10 + ["Good"] * 10):
        q = {"Poor": 0, "Medium": 1, "Good": 2}[quality]
        rows.append({"patient_id": f"p{i:02d}", "quality": quality, "a": q + i * 0.001, "b": 2 * q + i * 0.002, "noise": (i * 13) % 7})
    data = pd.DataFrame(rows)
    ranking = association_table(data, ["a", "b", "noise"])
    selected, rejected = select_nonredundant(data, ranking, ["a", "b", "noise"], threshold=0.90)
    assert "a" in selected or "b" in selected
    assert not ({"a", "b"} <= set(selected))
    assert len(rejected) >= 1


def test_transform_preserves_shape() -> None:
    image, _ = generate_echo_image("Medium", seed=12)
    spec = TransformationSpec("combo", (("median", {"radius": 1}), ("clahe", {"clip_limit": 0.01, "kernel_size": 16})))
    out = apply_transformation(image, spec)
    assert out.shape == image.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_reference_and_optimizer_return_valid_result() -> None:
    rows = []
    for seed in range(12):
        image, mask = generate_echo_image("Good", seed=seed)
        rows.append({"quality": "Good", **extract_features(image, mask)})
    for seed in range(12, 20):
        image, mask = generate_echo_image("Medium", seed=seed)
        rows.append({"quality": "Medium", **extract_features(image, mask)})
    data = pd.DataFrame(rows)
    feature_names = [
        "local_contrast_mean",
        "laplacian_variance",
        "tenengrad_mean",
        "noise_mad_highpass",
        "speckle_index_global",
        "glcm_contrast",
        "glcm_homogeneity",
        "cnr_myo_lv",
    ]
    ranking = association_table(data, feature_names)
    ref = GoodReference.fit(data, feature_names, association_ranking=ranking)
    medium, mask = generate_echo_image("Medium", seed=99)
    result = optimize_image(medium, mask, ref, gate_status="APTA_CONDICIONADA", max_candidates=12)
    assert len(result.candidates) >= 1
    assert result.best_image.shape == medium.shape
    assert result.recommendation
