from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr
from sklearn.feature_selection import mutual_info_classif

from echoquality.constants import QUALITY_ORDER, feature_level


def benjamini_hochberg(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    if p.size == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * p.size / np.arange(1, p.size + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    result = np.empty_like(q)
    result[order] = np.clip(q, 0.0, 1.0)
    return result


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
    """Exploratory ordinal associations for Good/Medium/Poor.

    This table is descriptive. It must not be used to preselect features before
    nested cross-validation. Supervised feature selection for the Quality Gate
    is performed inside each training fold.
    """

    if quality_column not in data:
        raise ValueError(f"Missing quality column: {quality_column}")
    quality = data[quality_column].map(QUALITY_ORDER)
    if quality.isna().any():
        bad = sorted(data.loc[quality.isna(), quality_column].astype(str).unique())
        raise ValueError(f"Unrecognized quality labels: {bad}")

    features = [column for column in feature_columns if column in data.columns]
    if not features:
        return pd.DataFrame()
    x_matrix = data[features].apply(pd.to_numeric, errors="coerce")
    medians = x_matrix.median(numeric_only=True).fillna(0.0)
    x_mi = x_matrix.fillna(medians).fillna(0.0).to_numpy(dtype=float)
    try:
        mi = mutual_info_classif(
            x_mi,
            quality.to_numpy(dtype=int),
            discrete_features=False,
            random_state=random_state,
        )
    except ValueError:
        mi = np.zeros(len(features), dtype=float)

    rows: list[dict[str, float | str | int]] = []
    y_all = quality.to_numpy(dtype=float)
    for index, name in enumerate(features):
        x = x_matrix[name].to_numpy(dtype=float)
        keep = np.isfinite(x) & np.isfinite(y_all)
        xx = x[keep]
        yy = y_all[keep].astype(int)
        if np.unique(xx).size < 3 or np.unique(yy).size < 2:
            rho, p_s = 0.0, 1.0
            h, p_kw, eps = 0.0, 1.0, 0.0
        else:
            rho, p_s = spearmanr(xx, yy)
            groups = [xx[yy == value] for value in sorted(np.unique(yy))]
            if min(len(group) for group in groups) >= 2:
                h, p_kw = kruskal(*groups)
                n, k = len(xx), len(groups)
                eps = max(0.0, float((h - k + 1) / max(1, n - k)))
            else:
                h, p_kw, eps = 0.0, 1.0, 0.0
        rows.append(
            {
                "feature": name,
                "level": feature_level(name) or "unknown",
                "n": int(keep.sum()),
                "spearman_rho": float(0.0 if not np.isfinite(rho) else rho),
                "spearman_p": float(1.0 if not np.isfinite(p_s) else p_s),
                "kruskal_h": float(h),
                "kruskal_p": float(p_kw),
                "epsilon_squared": float(eps),
                "fisher_score": fisher_multiclass(xx, yy) if keep.any() else 0.0,
                "mutual_information": float(mi[index]),
            }
        )
    result = pd.DataFrame(rows)
    result["spearman_q"] = benjamini_hochberg(result["spearman_p"])
    result["kruskal_q"] = benjamini_hochberg(result["kruskal_p"])
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
    """Bootstrap complete patients to quantify descriptive feature stability."""

    features = list(feature_columns)
    if not features:
        return StabilityResult(pd.DataFrame(), pd.DataFrame())
    rng = np.random.default_rng(seed)
    patients = np.asarray(sorted(data[patient_column].astype(str).unique()))
    counts: defaultdict[str, int] = defaultdict(int)
    sign_counts: defaultdict[str, int] = defaultdict(int)
    selected_rows: list[dict[str, object]] = []
    for bootstrap_index in range(int(n_bootstrap)):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        blocks: list[pd.DataFrame] = []
        for draw_index, patient in enumerate(sampled):
            block = data[data[patient_column].astype(str) == patient].copy()
            block[patient_column] = block[patient_column].astype(str) + f"__b{bootstrap_index}_{draw_index}"
            blocks.append(block)
        boot = pd.concat(blocks, ignore_index=True)
        # Bootstrap uses the predeclared Spearman/FDR criterion only.  The full
        # association table (Kruskal, Fisher, mutual information) is computed
        # once on the original sample; repeating those expensive descriptive
        # statistics does not change the bootstrap selection rule.
        x_boot = boot[features].apply(pd.to_numeric, errors="coerce")
        x_boot = x_boot.fillna(x_boot.median(numeric_only=True)).fillna(0.0)
        y_boot = boot[quality_column].map(QUALITY_ORDER).to_numpy(dtype=float)

        # Constant columns are common in small patient bootstraps and make a
        # correlation coefficient undefined. Score only variable features, then
        # restore constant columns as rho=0, p=1 without emitting misleading
        # numerical warnings.
        variable = x_boot.nunique(dropna=False) > 1
        variable_features = [name for name in features if bool(variable.get(name, False))]
        rho = np.zeros(len(features), dtype=float)
        p_s = np.ones(len(features), dtype=float)
        if variable_features and np.unique(y_boot).size > 1:
            matrix = np.column_stack([x_boot[variable_features].to_numpy(dtype=float), y_boot])
            correlation, pvalues = spearmanr(matrix, axis=0)
            if np.ndim(correlation) != 0:
                variable_rho = np.asarray(correlation[:-1, -1], dtype=float)
                variable_p = np.asarray(pvalues[:-1, -1], dtype=float)
                position = {name: index for index, name in enumerate(features)}
                for local_index, name in enumerate(variable_features):
                    rho[position[name]] = variable_rho[local_index]
                    p_s[position[name]] = variable_p[local_index]
        rho = np.where(np.isfinite(rho), rho, 0.0)
        p_s = np.where(np.isfinite(p_s), p_s, 1.0)
        q = benjamini_hochberg(p_s)
        ranking = pd.DataFrame({"feature": features, "spearman_rho": rho, "spearman_q": q})
        ranking = ranking.sort_values("spearman_rho", key=lambda values: values.abs(), ascending=False)
        eligible = ranking[
            (ranking["spearman_q"] <= alpha_fdr)
            & (ranking["spearman_rho"].abs() >= min_abs_rho)
        ].head(top_k)
        for _, row in eligible.iterrows():
            name = str(row["feature"])
            sign = int(np.sign(float(row["spearman_rho"])))
            counts[name] += 1
            sign_counts[name] += sign
            selected_rows.append(
                {
                    "bootstrap": bootstrap_index,
                    "feature": name,
                    "level": feature_level(name) or "unknown",
                    "spearman_rho": float(row["spearman_rho"]),
                }
            )
    rows = []
    for name in features:
        count = counts[name]
        rows.append(
            {
                "feature": name,
                "level": feature_level(name) or "unknown",
                "selection_frequency": count / max(1, int(n_bootstrap)),
                "sign_consistency": abs(sign_counts[name]) / max(1, count),
                "dominant_sign": int(np.sign(sign_counts[name])) if count else 0,
            }
        )
    frequencies = pd.DataFrame(rows).sort_values(
        ["selection_frequency", "sign_consistency"], ascending=False
    ).reset_index(drop=True)
    return StabilityResult(frequencies, pd.DataFrame(selected_rows))


def select_nonredundant(
    data: pd.DataFrame,
    ranking: pd.DataFrame,
    candidates: Iterable[str],
    *,
    threshold: float = 0.90,
) -> tuple[list[str], pd.DataFrame]:
    candidate_set = set(candidates)
    ordered = [name for name in ranking["feature"].astype(str) if name in candidate_set]
    if not ordered:
        return [], pd.DataFrame(columns=["feature_rejected", "representative_kept", "abs_spearman"])
    correlation = data[ordered].apply(pd.to_numeric, errors="coerce").corr(method="spearman").abs()
    selected: list[str] = []
    rejected: list[dict[str, object]] = []
    for name in ordered:
        conflict: str | None = None
        value = 0.0
        for kept in selected:
            rho = float(correlation.loc[name, kept]) if name in correlation.index and kept in correlation.columns else 0.0
            if rho >= threshold:
                conflict, value = kept, rho
                break
        if conflict is None:
            selected.append(name)
        else:
            rejected.append(
                {
                    "feature_rejected": name,
                    "representative_kept": conflict,
                    "abs_spearman": value,
                }
            )
    return selected, pd.DataFrame(rejected)


def discover_indicators(
    data: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    bootstrap: int = 200,
    top_k: int = 15,
    min_frequency: float = 0.65,
    min_sign_consistency: float = 0.80,
    redundancy: float = 0.90,
    seed: int = 2026,
) -> dict[str, pd.DataFrame]:
    """Run exploratory indicator discovery and explicitly record fallback use."""

    features = list(feature_columns)
    ranking = association_table(data, features, random_state=seed)
    stability = bootstrap_stability(
        data,
        features,
        n_bootstrap=bootstrap,
        top_k=top_k,
        seed=seed,
    )
    merged = ranking.merge(stability.frequencies, on=["feature", "level"], how="left")
    stable = merged[
        (merged["selection_frequency"] >= min_frequency)
        & (merged["sign_consistency"] >= min_sign_consistency)
    ].copy()
    if stable.empty:
        stable = merged.head(top_k).copy()
        stable["selection_reason"] = "fallback_top_k_no_feature_met_stability_thresholds"
        fallback_used = True
    else:
        stable["selection_reason"] = "met_stability_and_sign_thresholds"
        fallback_used = False
    selected, rejected = select_nonredundant(
        data,
        ranking,
        stable["feature"].astype(str),
        threshold=redundancy,
    )
    final = stable[stable["feature"].isin(selected)].copy()
    final["fallback_used"] = fallback_used
    final["exploratory_only"] = True
    final["warning"] = (
        "Do not feed this list into nested-CV evaluation. Feature selection for the Quality Gate is fitted inside each training fold."
    )
    return {
        "association_ranking": ranking,
        "feature_stability": stability.frequencies,
        "bootstrap_selections": stability.selections,
        "redundancy_rejections": rejected,
        "stable_nonredundant_indicators": final,
    }
