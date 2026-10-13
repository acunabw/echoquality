from __future__ import annotations

import numpy as np
import pandas as pd

from echoquality.recognition.good_reference import build_good_references


def test_good_reference_is_level_and_view_specific() -> None:
    rng = np.random.default_rng(4)
    rows = []
    for patient in range(12):
        for view in ("2CH", "4CH"):
            rows.append(
                {
                    "patient_id": f"p{patient:03d}",
                    "view": view,
                    "quality": "Good",
                    "not_apt": 0,
                    "global__a__mean": rng.normal(),
                    "anatomical__a__mean": rng.normal(),
                    "textural__a__mean": rng.normal(),
                }
            )
    data = pd.DataFrame(rows)
    bundle = build_good_references(data)
    for level in ("global", "anatomical", "textural"):
        assert bundle.get(level, "2CH").n_good == 12
        assert np.isfinite(bundle.get(level, "2CH").distance_q95)
