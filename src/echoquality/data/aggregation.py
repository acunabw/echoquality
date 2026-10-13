from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from echoquality.constants import META_COLUMNS


def _phase_value(group: pd.DataFrame, phase: str, feature: str) -> float:
    values = pd.to_numeric(group.loc[group["phase"].astype(str).str.upper() == phase, feature], errors="coerce")
    return float(values.iloc[0]) if len(values) else float("nan")


def aggregate_patient_view(
    frame_features: pd.DataFrame,
    feature_columns: Iterable[str] | None = None,
    *,
    modes: tuple[str, ...] = ("mean", "abs_delta"),
    phases: tuple[str, str] = ("ED", "ES"),
) -> pd.DataFrame:
    """Aggregate frame features while retaining explicit phase-disagreement information.

    Unlike the v1 table, the output names the aggregation operation. With exactly two
    phases, the median equals the arithmetic mean and can hide a phase-specific defect;
    ``abs_delta`` is therefore included by default.
    """

    required = {"patient_id", "view", "phase", "quality"}
    missing = required - set(frame_features.columns)
    if missing:
        raise ValueError(f"Missing aggregation columns: {sorted(missing)}")
    if feature_columns is None:
        feature_columns = [column for column in frame_features.columns if column not in META_COLUMNS]
    features = list(feature_columns)
    allowed = {"mean", "abs_delta", "ed", "es", "min", "max"}
    unknown = set(modes) - allowed
    if unknown:
        raise ValueError(f"Unknown aggregation modes: {sorted(unknown)}")

    rows: list[dict[str, object]] = []
    group_columns = ["patient_id", "view", "quality"]
    for keys, group in frame_features.groupby(group_columns, sort=True, dropna=False):
        row: dict[str, object] = dict(zip(group_columns, keys))
        row["not_apt"] = int(str(row["quality"]) == "Poor")
        row["apt_target"] = 1 - int(row["not_apt"])
        row["phase_count"] = int(group["phase"].nunique())
        for feature in features:
            values = pd.to_numeric(group[feature], errors="coerce").to_numpy(dtype=float)
            if "mean" in modes:
                row[f"{feature}__mean"] = float(np.nanmean(values))
            if "min" in modes:
                row[f"{feature}__min"] = float(np.nanmin(values))
            if "max" in modes:
                row[f"{feature}__max"] = float(np.nanmax(values))
            ed = _phase_value(group, phases[0].upper(), feature)
            es = _phase_value(group, phases[1].upper(), feature)
            if "ed" in modes:
                row[f"{feature}__ed"] = ed
            if "es" in modes:
                row[f"{feature}__es"] = es
            if "abs_delta" in modes:
                row[f"{feature}__abs_delta"] = float(abs(ed - es)) if np.isfinite(ed) and np.isfinite(es) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
