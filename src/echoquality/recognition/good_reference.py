from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from echoquality.constants import LEVELS
from echoquality.recognition.hierarchical_gate import infer_feature_columns_by_level


@dataclass
class LevelReference:
    level: str
    stratum: str
    feature_names: list[str]
    median: np.ndarray
    mad_scale: np.ndarray
    q10: np.ndarray
    q90: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    distance_q95: float
    n_good: int

    def transform(self, values: pd.Series | dict[str, Any]) -> np.ndarray:
        vector = np.asarray([float(values.get(name, np.nan)) for name in self.feature_names], dtype=float)
        vector = np.where(np.isfinite(vector), vector, self.median)
        return (vector - self.median) / self.mad_scale

    def robust_z(self, values: pd.Series | dict[str, Any]) -> dict[str, float]:
        z = self.transform(values)
        return {name: float(value) for name, value in zip(self.feature_names, z)}

    def mahalanobis(self, values: pd.Series | dict[str, Any]) -> float:
        z = self.transform(values)
        return float(np.sqrt(max(0.0, z @ self.precision @ z)))


@dataclass
class GoodReferenceBundle:
    references: dict[str, dict[str, LevelReference]] = field(default_factory=dict)
    quality_label: str = "Good"
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, level: str, view: str | None = None) -> LevelReference:
        level_refs = self.references[level]
        if view is not None and str(view) in level_refs:
            return level_refs[str(view)]
        return level_refs["ALL"]


def _fit_reference(frame: pd.DataFrame, features: list[str], *, level: str, stratum: str) -> LevelReference:
    numeric = frame[features].apply(pd.to_numeric, errors="coerce")
    usable = [
        column
        for column in features
        if numeric[column].notna().sum() >= 3 and numeric[column].nunique(dropna=True) > 1
    ]
    if not usable:
        raise ValueError(f"No usable Good-reference features for {level}/{stratum}")
    x = numeric[usable].to_numpy(dtype=float)
    median = np.nanmedian(x, axis=0)
    x = np.where(np.isfinite(x), x, median)
    raw_mad = np.nanmedian(np.abs(x - median), axis=0)
    fallback_scale = np.nanstd(x, axis=0, ddof=1)
    mad_scale = 1.4826 * raw_mad
    mad_scale = np.where(mad_scale > 1e-8, mad_scale, np.where(fallback_scale > 1e-8, fallback_scale, 1.0))
    z = (x - median) / mad_scale
    if z.shape[0] >= 3:
        fitted = LedoitWolf().fit(z)
        covariance = fitted.covariance_
        precision = fitted.precision_
    else:
        covariance = np.eye(z.shape[1])
        precision = np.eye(z.shape[1])
    distances = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", z, precision, z)))
    return LevelReference(
        level=level,
        stratum=stratum,
        feature_names=usable,
        median=median,
        mad_scale=mad_scale,
        q10=np.quantile(x, 0.10, axis=0),
        q90=np.quantile(x, 0.90, axis=0),
        covariance=covariance,
        precision=precision,
        distance_q95=float(np.quantile(distances, 0.95)),
        n_good=int(len(frame)),
    )


def build_good_references(
    data: pd.DataFrame,
    *,
    quality_column: str = "quality",
    good_label: str = "Good",
    view_column: str = "view",
    stratify_by_view: bool = True,
) -> GoodReferenceBundle:
    """Build separate robust representation references for each level and view."""

    good = data[data[quality_column].astype(str) == good_label].copy()
    if good.empty:
        raise ValueError("No Good observations are available to build references")
    columns_by_level = infer_feature_columns_by_level(good)
    references: dict[str, dict[str, LevelReference]] = {}
    for level in LEVELS:
        features = columns_by_level[level]
        references[level] = {"ALL": _fit_reference(good, features, level=level, stratum="ALL")}
        if stratify_by_view and view_column in good:
            for view, group in good.groupby(view_column):
                if len(group) >= 5:
                    references[level][str(view)] = _fit_reference(
                        group,
                        features,
                        level=level,
                        stratum=str(view),
                    )
    return GoodReferenceBundle(
        references=references,
        quality_label=good_label,
        metadata={
            "n_good": int(len(good)),
            "n_good_patients": int(good["patient_id"].nunique()) if "patient_id" in good else None,
            "stratify_by_view": stratify_by_view,
            "reference_statistics": "median_MAD_q10_q90_LedoitWolf_covariance",
        },
    )


def reference_summary(bundle: GoodReferenceBundle) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for level, strata in bundle.references.items():
        for stratum, reference in strata.items():
            for index, feature in enumerate(reference.feature_names):
                rows.append(
                    {
                        "level": level,
                        "stratum": stratum,
                        "feature": feature,
                        "median": reference.median[index],
                        "mad_scale": reference.mad_scale[index],
                        "q10": reference.q10[index],
                        "q90": reference.q90[index],
                        "distance_q95": reference.distance_q95,
                        "n_good": reference.n_good,
                    }
                )
    return pd.DataFrame(rows)


def save_references(bundle: GoodReferenceBundle, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    reference_summary(bundle).to_csv(path.with_suffix(".csv"), index=False)
    metadata = {
        "quality_label": bundle.quality_label,
        "metadata": bundle.metadata,
        "levels": {
            level: {
                stratum: {
                    "n_good": reference.n_good,
                    "n_features": len(reference.feature_names),
                    "distance_q95": reference.distance_q95,
                }
                for stratum, reference in strata.items()
            }
            for level, strata in bundle.references.items()
        },
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_references(path: str | Path) -> GoodReferenceBundle:
    bundle = joblib.load(path)
    if not isinstance(bundle, GoodReferenceBundle):
        raise TypeError(f"Unexpected Good Reference artifact: {type(bundle)}")
    return bundle
