from __future__ import annotations

import pandas as pd
import pytest

from echoquality.data.manifests import select_patient_subset


def test_nested_reproducible_patient_subsets(tmp_path) -> None:
    rows = []
    for patient in range(20):
        for view in ("2CH", "4CH"):
            rows.append(
                {
                    "patient_id": f"p{patient:03d}",
                    "view": view,
                    "phase": "ED",
                    "quality": "Good",
                    "image_path": "dummy",
                }
            )
    manifest = pd.DataFrame(rows)
    order_file = tmp_path / "order.txt"
    subset5, selected5 = select_patient_subset(manifest, 5, seed=8, order_file=order_file)
    subset10, selected10 = select_patient_subset(manifest, 10, seed=999, order_file=order_file)
    assert selected10[:5] == selected5
    assert subset5["patient_id"].nunique() == 5
    assert subset10["patient_id"].nunique() == 10


def test_patient_order_rejects_a_different_dataset(tmp_path) -> None:
    manifest = pd.DataFrame(
        [
            {
                "patient_id": "camus001",
                "view": "2CH",
                "phase": "ED",
                "quality": "Good",
                "image_path": "dummy",
            },
            {
                "patient_id": "camus002",
                "view": "2CH",
                "phase": "ED",
                "quality": "Poor",
                "image_path": "dummy",
            },
        ]
    )
    order_file = tmp_path / "order.txt"
    order_file.write_text("demo_001\ndemo_002\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match this dataset"):
        select_patient_subset(manifest, 1, seed=2026, order_file=order_file)
