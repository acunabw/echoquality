from __future__ import annotations

import numpy as np
import pandas as pd

from echoquality.recognition.hierarchical_gate import nested_group_validation


def _data() -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    rows = []
    for patient_index in range(18):
        target = int(patient_index < 6)
        for view in ("2CH", "4CH"):
            row = {
                "patient_id": f"p{patient_index:03d}",
                "view": view,
                "quality": "Poor" if target else "Good",
                "not_apt": target,
                "apt_target": 1 - target,
            }
            for level, effect in (("global", 0.7), ("anatomical", 0.5), ("textural", 0.9)):
                for feature_index in range(3):
                    row[f"{level}__f{feature_index}__mean"] = rng.normal(target * effect, 1.0)
                    row[f"{level}__f{feature_index}__abs_delta"] = abs(rng.normal(target * 0.15, 0.25))
            rows.append(row)
    return pd.DataFrame(rows)


def test_late_fusion_nested_group_validation_has_no_patient_overlap() -> None:
    result = nested_group_validation(
        _data(),
        model_names=("LogisticRegression",),
        outer_splits=2,
        inner_splits=2,
        fast=True,
    )
    assert not result.summary.empty
    assert (result.fold_metrics["patient_overlap"] == 0).all()
    assert {"score_global", "score_anatomical", "score_textural", "probability_not_apt"}.issubset(result.predictions)
