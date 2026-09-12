from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


@dataclass
class GoodReference:
    """Patron multivariado aprendido exclusivamente con imagenes Good de entrenamiento."""

    feature_names: list[str]
    median: np.ndarray
    scale: np.ndarray
    q10: np.ndarray
    q90: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    precision: np.ndarray
    direction: np.ndarray
    n_good: int

    @classmethod
    def fit(
        cls,
        data: pd.DataFrame,
        feature_names: list[str],
        *,
        quality_column: str = "quality",
        association_ranking: pd.DataFrame | None = None,
    ) -> "GoodReference":
        good = data[data[quality_column].astype(str) == "Good"]
        if len(good) < 5:
            raise ValueError("Se requieren al menos cinco observaciones Good")
        x = good[feature_names].apply(pd.to_numeric, errors="coerce")
        medians_for_imputation = x.median()
        x = x.fillna(medians_for_imputation)
        arr = x.to_numpy(dtype=float)
        median = np.median(arr, axis=0)
        mad = np.median(np.abs(arr - median), axis=0)
        std = np.std(arr, axis=0, ddof=1)
        scale = 1.4826 * mad
        scale = np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.0))
        q10, q90 = np.quantile(arr, [0.10, 0.90], axis=0)
        lw = LedoitWolf().fit((arr - median) / scale)
        direction = np.zeros(len(feature_names), dtype=int)
        if association_ranking is not None and not association_ranking.empty:
            mapping = association_ranking.set_index("feature")["spearman_rho"].to_dict()
            direction = np.array([int(np.sign(float(mapping.get(name, 0.0)))) for name in feature_names])
        return cls(
            feature_names=list(feature_names),
            median=median,
            scale=scale,
            q10=q10,
            q90=q90,
            mean=np.mean(arr, axis=0),
            covariance=lw.covariance_,
            precision=lw.precision_,
            direction=direction,
            n_good=len(good),
        )

    def vector(self, features: dict[str, float] | pd.Series) -> np.ndarray:
        return np.array([float(features.get(name, np.nan)) for name in self.feature_names], dtype=float)

    def _impute(self, vector: np.ndarray) -> np.ndarray:
        x = np.asarray(vector, dtype=float).copy()
        missing = ~np.isfinite(x)
        x[missing] = self.median[missing]
        return x

    def robust_z(self, features: dict[str, float] | pd.Series) -> np.ndarray:
        x = self._impute(self.vector(features))
        return (x - self.median) / self.scale

    def mahalanobis(self, features: dict[str, float] | pd.Series) -> float:
        z = self.robust_z(features)
        return float(np.sqrt(max(0.0, z @ self.precision @ z)))

    def deviation_table(self, features: dict[str, float] | pd.Series) -> pd.DataFrame:
        x = self._impute(self.vector(features))
        z = (x - self.median) / self.scale
        rows: list[dict[str, Any]] = []
        for i, name in enumerate(self.feature_names):
            direction = int(self.direction[i])
            if direction > 0:
                deficiency = max(0.0, -float(z[i]))
                expected = "alto_en_Good"
            elif direction < 0:
                deficiency = max(0.0, float(z[i]))
                expected = "bajo_en_Good"
            else:
                if x[i] < self.q10[i]:
                    deficiency = float((self.q10[i] - x[i]) / self.scale[i])
                elif x[i] > self.q90[i]:
                    deficiency = float((x[i] - self.q90[i]) / self.scale[i])
                else:
                    deficiency = 0.0
                expected = "dentro_del_intervalo_Good"
            rows.append(
                {
                    "feature": name,
                    "value": float(x[i]),
                    "good_median": float(self.median[i]),
                    "good_q10": float(self.q10[i]),
                    "good_q90": float(self.q90[i]),
                    "robust_z": float(z[i]),
                    "association_direction": direction,
                    "expected_behavior": expected,
                    "deficiency": float(deficiency),
                    "outside_good_interval": bool(x[i] < self.q10[i] or x[i] > self.q90[i]),
                }
            )
        return pd.DataFrame(rows).sort_values("deficiency", ascending=False).reset_index(drop=True)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "GoodReference":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError("El archivo no contiene GoodReference")
        return obj
