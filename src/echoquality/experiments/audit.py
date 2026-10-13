from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from echoquality.constants import META_COLUMNS, TARGET_COLUMNS, feature_level


@dataclass
class AuditResult:
    checks: pd.DataFrame
    passed: bool


def _check(name: str, passed: bool, value: Any, expected: str, severity: str = "error") -> dict[str, Any]:
    return {
        "check": name,
        "passed": bool(passed),
        "value": value,
        "expected": expected,
        "severity": severity,
    }


def audit_feature_tables(frame: pd.DataFrame, view: pd.DataFrame) -> AuditResult:
    rows: list[dict[str, Any]] = []
    feature_columns = [column for column in frame.columns if feature_level(str(column))]
    target_leakage = sorted(set(feature_columns).intersection(TARGET_COLUMNS))
    rows.append(_check("no_targets_in_feature_matrix", not target_leakage, target_leakage, "[]"))
    metadata_leakage = sorted(set(feature_columns).intersection(META_COLUMNS))
    rows.append(_check("no_metadata_in_feature_matrix", not metadata_leakage, metadata_leakage, "[]"))
    rows.append(_check("frame_features_have_three_levels", {feature_level(c) for c in feature_columns} >= {"global", "anatomical", "textural"}, sorted({feature_level(c) for c in feature_columns}), "global, anatomical, textural"))

    if {"global__intensity_p95", "global__intensity_p05", "global__robust_intensity_range_p95_p05"}.issubset(frame):
        error = np.nanmax(
            np.abs(
                frame["global__robust_intensity_range_p95_p05"].to_numpy(dtype=float)
                - (frame["global__intensity_p95"] - frame["global__intensity_p05"]).to_numpy(dtype=float)
            )
        )
        rows.append(_check("robust_range_identity", error < 1e-10, float(error), "max_abs_error < 1e-10"))
    if {"global__speckle_index", "global__intensity_std", "global__brightness_mean"}.issubset(frame):
        expected = frame["global__intensity_std"] / frame["global__brightness_mean"].abs().clip(lower=1e-8)
        error = float(np.nanmax(np.abs(frame["global__speckle_index"] - expected)))
        rows.append(_check("global_speckle_identity", error < 1e-8, error, "max_abs_error < 1e-8"))
    if {"anatomical__mask_foreground_fraction", "anatomical__lv_area_fraction", "anatomical__myo_area_fraction", "anatomical__la_area_fraction"}.issubset(frame):
        expected = frame["anatomical__lv_area_fraction"] + frame["anatomical__myo_area_fraction"] + frame["anatomical__la_area_fraction"]
        error = float(np.nanmax(np.abs(frame["anatomical__mask_foreground_fraction"] - expected)))
        rows.append(_check("mask_fraction_identity", error < 1e-10, error, "max_abs_error < 1e-10"))
    if {"textural__glcm_asm_uniformity", "textural__glcm_energy_sqrt_asm"}.issubset(frame):
        error = float(np.nanmax(np.abs(frame["textural__glcm_energy_sqrt_asm"] ** 2 - frame["textural__glcm_asm_uniformity"])))
        rows.append(_check("glcm_energy_squared_equals_asm", error < 1e-8, error, "max_abs_error < 1e-8"))

    # Validate explicit aggregation; unlike v1, operation names are retained.
    merged_errors: list[float] = []
    group_keys = ["patient_id", "view", "quality"]
    for base in feature_columns[: min(20, len(feature_columns))]:
        mean_column = f"{base}__mean"
        delta_column = f"{base}__abs_delta"
        if mean_column not in view:
            continue
        expected_mean = frame.groupby(group_keys)[base].mean().rename("expected").reset_index()
        observed = view[group_keys + [mean_column]].merge(expected_mean, on=group_keys, how="left")
        merged_errors.append(float(np.nanmax(np.abs(observed[mean_column] - observed["expected"]))))
        if delta_column in view and "phase" in frame:
            pivot = frame.pivot_table(index=group_keys, columns="phase", values=base, aggfunc="first").reset_index()
            if {"ED", "ES"}.issubset(pivot):
                pivot["expected_delta"] = (pivot["ED"] - pivot["ES"]).abs()
                observed_delta = view[group_keys + [delta_column]].merge(pivot[group_keys + ["expected_delta"]], on=group_keys, how="left")
                merged_errors.append(float(np.nanmax(np.abs(observed_delta[delta_column] - observed_delta["expected_delta"]))))
    aggregation_error = max(merged_errors) if merged_errors else float("nan")
    rows.append(_check("explicit_temporal_aggregation", np.isfinite(aggregation_error) and aggregation_error < 1e-10, aggregation_error, "mean and abs_delta columns reproduce ED/ES calculations"))

    checks = pd.DataFrame(rows)
    failed_errors = checks[(~checks["passed"]) & (checks["severity"] == "error")]
    return AuditResult(checks=checks, passed=failed_errors.empty)


def audit_manifest(manifest: pd.DataFrame) -> AuditResult:
    rows = []
    required = {"patient_id", "view", "phase", "quality", "not_apt", "apt_target", "image_path", "mask_path"}
    rows.append(_check("manifest_required_columns", required.issubset(manifest.columns), sorted(required - set(manifest.columns)), "no missing required columns"))
    duplicate_count = int(manifest.duplicated(["patient_id", "view", "phase"]).sum())
    rows.append(_check("unique_patient_view_phase", duplicate_count == 0, duplicate_count, "0"))
    target_ok = bool(((manifest["quality"] == "Poor").astype(int) == manifest["not_apt"].astype(int)).all()) if {"quality", "not_apt"}.issubset(manifest) else False
    rows.append(_check("target_encoding_consistent", target_ok, target_ok, "Poor=not_apt=1; Good/Medium=0"))
    checks = pd.DataFrame(rows)
    return AuditResult(checks, bool(checks["passed"].all()))
