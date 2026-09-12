from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from sklearn.feature_selection import mutual_info_classif


QUALITY_ORDER = {"Poor": 0, "Medium": 1, "Good": 2}


def benjamini_hochberg(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def fisher_multiclass(values: np.ndarray, labels: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    y = np.asarray(labels)
    overall = float(np.nanmean(x))
    between = 0.0
    within = 0.0
    for label in np.unique(y):
        group = x[y == label]
        group = group[np.isfinite(group)]
        if group.size == 0:
            continue
        between += group.size * (float(group.mean()) - overall) ** 2
        within += float(np.sum((group - group.mean()) ** 2))
    return float(between / (within + 1e-12))


def association_table(
    data: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    quality_column: str = "quality",
    random_state: int = 2026,
) -> pd.DataFrame:
    """Calcula asociacion ordinal y separacion entre Good/Medium/Poor."""

    if quality_column not in data:
        raise ValueError(f"Falta la columna {quality_column}")
    quality = data[quality_column].map(QUALITY_ORDER)
    if quality.isna().any():
        bad = sorted(data.loc[quality.isna(), quality_column].astype(str).unique())
        raise ValueError(f"Etiquetas de calidad no reconocidas: {bad}")

    features = [c for c in feature_columns if c in data.columns]
    x_matrix = data[features].apply(pd.to_numeric, errors="coerce")
    x_mi = x_matrix.fillna(x_matrix.median()).to_numpy(dtype=float)
    mi = mutual_info_classif(x_mi, quality.to_numpy(), discrete_features=False, random_state=random_state)

    rows: list[dict[str, float | str | int]] = []
    for idx, name in enumerate(features):
        x = x_matrix[name].to_numpy(dtype=float)
        keep = np.isfinite(x) & np.isfinite(quality.to_numpy(dtype=float))
        xx = x[keep]
        yy = quality.to_numpy()[keep]
        if np.unique(xx).size < 3 or np.unique(yy).size < 2:
            rho, p_s = 0.0, 1.0
            h, p_kw = 0.0, 1.0
            eps = 0.0
        else:
            rho, p_s = spearmanr(xx, yy)
            groups = [xx[yy == value] for value in sorted(np.unique(yy))]
            if min(len(g) for g in groups) >= 2:
                h, p_kw = kruskal(*groups)
                n = len(xx)
                k = len(groups)
                eps = max(0.0, float((h - k + 1) / max(1, n - k)))
            else:
                h, p_kw, eps = 0.0, 1.0, 0.0
        rows.append(
            {
                "feature": name,
                "n": int(keep.sum()),
                "spearman_rho": float(0.0 if not np.isfinite(rho) else rho),
                "spearman_p": float(1.0 if not np.isfinite(p_s) else p_s),
                "kruskal_h": float(h),
                "kruskal_p": float(p_kw),
                "epsilon_squared": float(eps),
                "fisher_score": fisher_multiclass(xx, yy) if keep.sum() else 0.0,
                "mutual_information": float(mi[idx]),
            }
        )
    result = pd.DataFrame(rows)
    result["spearman_q"] = benjamini_hochberg(result["spearman_p"])
    result["kruskal_q"] = benjamini_hochberg(result["kruskal_p"])
    # Puntaje compuesto solo para ranking; no sustituye la inferencia estadistica.
    result["association_score"] = (
        result["spearman_rho"].abs().rank(pct=True)
        + result["epsilon_squared"].rank(pct=True)
        + result["fisher_score"].rank(pct=True)
        + result["mutual_information"].rank(pct=True)
    ) / 4.0
    return result.sort_values(["association_score", "spearman_rho"], ascending=[False, False]).reset_index(drop=True)


@dataclass
class StabilityResult:
    frequencies: pd.DataFrame
    selections: pd.DataFrame


def bootstrap_stability(
    data: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    patient_column: str = "patient_id",
    quality_column: str = "quality",
    n_bootstrap: int = 200,
    top_k: int = 15,
    min_abs_rho: float = 0.20,
    alpha_fdr: float = 0.05,
    seed: int = 2026,
) -> StabilityResult:
    """Repite la seleccion remuestreando pacientes completos."""

    rng = np.random.default_rng(seed)
    patients = np.asarray(sorted(data[patient_column].astype(str).unique()))
    counts: defaultdict[str, int] = defaultdict(int)
    sign_counts: defaultdict[str, int] = defaultdict(int)
    selected_rows: list[dict[str, object]] = []
    feature_columns = list(feature_columns)
    for b in range(n_bootstrap):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        blocks = []
        for draw_idx, patient in enumerate(sampled):
            block = data[data[patient_column].astype(str) == patient].copy()
            # ID artificial evita que duplicados bootstrap se fusionen accidentalmente.
            block[patient_column] = block[patient_column].astype(str) + f"__b{b}_{draw_idx}"
            blocks.append(block)
        boot = pd.concat(blocks, ignore_index=True)
        ranking = association_table(boot, feature_columns, quality_column=quality_column, random_state=seed + b)
        eligible = ranking[
            (ranking["spearman_q"] <= alpha_fdr)
            & (ranking["spearman_rho"].abs() >= min_abs_rho)
        ].head(top_k)
        for _, row in eligible.iterrows():
            feature_name = str(row["feature"])
            sign = int(np.sign(float(row["spearman_rho"])))
            counts[feature_name] += 1
            sign_counts[feature_name] += sign
            selected_rows.append(
                {"bootstrap": b, "feature": feature_name, "spearman_rho": float(row["spearman_rho"])}
            )
    rows = []
    for feature_name in feature_columns:
        count = counts[feature_name]
        rows.append(
            {
                "feature": feature_name,
                "selection_frequency": count / max(1, n_bootstrap),
                "sign_consistency": abs(sign_counts[feature_name]) / max(1, count),
                "dominant_sign": int(np.sign(sign_counts[feature_name])) if count else 0,
            }
        )
    freq = pd.DataFrame(rows).sort_values("selection_frequency", ascending=False).reset_index(drop=True)
    return StabilityResult(freq, pd.DataFrame(selected_rows))


def select_nonredundant(
    data: pd.DataFrame,
    ranking: pd.DataFrame,
    candidates: Iterable[str],
    *,
    threshold: float = 0.90,
) -> tuple[list[str], pd.DataFrame]:
    """Seleccion greedy: conserva el mejor indicador de cada grupo correlacionado."""

    ordered = [f for f in ranking["feature"].astype(str) if f in set(candidates)]
    corr = data[ordered].apply(pd.to_numeric, errors="coerce").corr(method="spearman").abs()
    selected: list[str] = []
    rejected_rows: list[dict[str, object]] = []
    for feature_name in ordered:
        conflict = None
        conflict_r = 0.0
        for kept in selected:
            value = float(corr.loc[feature_name, kept]) if feature_name in corr.index and kept in corr.columns else 0.0
            if value >= threshold:
                conflict, conflict_r = kept, value
                break
        if conflict is None:
            selected.append(feature_name)
        else:
            rejected_rows.append(
                {
                    "feature_rejected": feature_name,
                    "representative_kept": conflict,
                    "abs_spearman": conflict_r,
                }
            )
    return selected, pd.DataFrame(rejected_rows)
