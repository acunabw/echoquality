from __future__ import annotations

import pandas as pd

from echoquality.constants import META_COLUMNS


_TEXTURAL_EXACT = {
    "lbp_entropy",
    "glcm_contrast",
    "glcm_dissimilarity",
    "glcm_homogeneity",
    "glcm_energy",
    "glcm_correlation",
    "glcm_asm",
    "glcm_entropy",
    "myo_speckle_index",
    "dropout_fraction_myo",
    "myo_entropy",
}
_ANATOMICAL_EXACT = {
    "mask_foreground_fraction",
    "myo_boundary_strength",
    "lv_boundary_strength",
    "cnr_myo_lv",
}


def legacy_feature_level(name: str) -> str:
    if name in _TEXTURAL_EXACT or name.startswith("glcm_") or name.startswith("lbp_"):
        return "textural"
    if name in _ANATOMICAL_EXACT:
        return "anatomical"
    if name.startswith(("lv_", "myo_", "la_")):
        return "anatomical"
    return "global"


def convert_v1_view_features(data: pd.DataFrame) -> pd.DataFrame:
    """Prefix v1 features by representation level for architectural ablation.

    This adapter does not manufacture the new ROI-only radiomics descriptors and
    must not be confused with a full v2 re-extraction from CAMUS images. It is
    useful only for comparing early fusion and late fusion on the exact legacy
    measurements.
    """

    metadata = [column for column in data.columns if column in META_COLUMNS or column in {"patient_id", "view", "quality", "not_apt", "apt_target"}]
    result = data[metadata].copy()
    for column in data.columns:
        if column in metadata:
            continue
        numeric = pd.to_numeric(data[column], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        level = legacy_feature_level(str(column))
        result[f"{level}__legacy__{column}__mean"] = numeric
    return result
