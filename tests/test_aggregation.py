from __future__ import annotations

import pandas as pd

from echoquality.data.aggregation import aggregate_patient_view


def test_temporal_aggregation_is_explicit() -> None:
    frame = pd.DataFrame(
        [
            {"patient_id": "p1", "view": "2CH", "phase": "ED", "quality": "Good", "global__x": 2.0},
            {"patient_id": "p1", "view": "2CH", "phase": "ES", "quality": "Good", "global__x": 6.0},
        ]
    )
    view = aggregate_patient_view(frame, ["global__x"])
    assert view.loc[0, "global__x__mean"] == 4.0
    assert view.loc[0, "global__x__abs_delta"] == 4.0
    assert view.loc[0, "not_apt"] == 0
