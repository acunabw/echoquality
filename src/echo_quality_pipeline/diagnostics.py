from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd


ISSUE_RULES: dict[str, tuple[str, ...]] = {
    "contrast": ("contrast", "dynamic_range", "cnr", "boundary_strength", "glcm_contrast"),
    "sharpness": ("laplacian", "tenengrad", "edge_strength", "edge_density"),
    "noise_speckle": ("noise", "speckle", "high_frequency"),
    "brightness_dynamic_range": ("intensity_mean", "intensity_p05", "intensity_p95", "saturation"),
    "dropout_signal_loss": ("dropout",),
    "texture": ("glcm", "lbp", "entropy", "local_variance"),
    "morphology_acquisition": ("mask_", "lv_", "myo_", "la_", "border_touch", "components", "solidity"),
}


@dataclass
class DiagnosticResult:
    indicator_table: pd.DataFrame
    issue_table: pd.DataFrame
    top_issues: list[str]
    non_correctable: bool
    action: str


def _issue_for_feature(feature_name: str) -> str:
    lower = feature_name.lower()
    for issue, tokens in ISSUE_RULES.items():
        if any(token in lower for token in tokens):
            return issue
    return "other"


def diagnose_deviations(
    deviations: pd.DataFrame,
    *,
    deficiency_threshold: float = 0.75,
    max_indicators: int = 12,
) -> DiagnosticResult:
    """Agrupa indicadores desviados en causas operativas interpretables."""

    table = deviations.copy()
    table["issue"] = table["feature"].astype(str).map(_issue_for_feature)
    flagged = table[table["deficiency"] >= deficiency_threshold].head(max_indicators)
    issue_score: defaultdict[str, float] = defaultdict(float)
    issue_count: defaultdict[str, int] = defaultdict(int)
    top_feature: dict[str, str] = {}
    for _, row in flagged.iterrows():
        issue = str(row["issue"])
        issue_score[issue] += float(row["deficiency"])
        issue_count[issue] += 1
        top_feature.setdefault(issue, str(row["feature"]))
    issues = pd.DataFrame(
        [
            {
                "issue": issue,
                "severity_score": score,
                "n_indicators": issue_count[issue],
                "top_indicator": top_feature[issue],
            }
            for issue, score in issue_score.items()
        ]
    )
    if not issues.empty:
        issues = issues.sort_values("severity_score", ascending=False).reset_index(drop=True)
    top_issues = issues["issue"].head(3).tolist() if not issues.empty else []
    severe_morphology = bool(
        not issues.empty
        and ((issues["issue"] == "morphology_acquisition") & (issues["severity_score"] >= 3.0)).any()
    )
    severe_dropout = bool(
        not issues.empty
        and ((issues["issue"] == "dropout_signal_loss") & (issues["severity_score"] >= 3.0)).any()
    )
    non_correctable = severe_morphology or severe_dropout
    if non_correctable:
        action = "REPETIR_ADQUISICION_O_REVISION_EXPERTA"
    elif top_issues:
        action = "EVALUAR_TRANSFORMACIONES_CANDIDATAS"
    else:
        action = "MANTENER_IMAGEN_ORIGINAL"
    return DiagnosticResult(table, issues, top_issues, non_correctable, action)


def diagnostic_summary(result: DiagnosticResult) -> dict[str, Any]:
    return {
        "top_issues": result.top_issues,
        "non_correctable": result.non_correctable,
        "action": result.action,
        "n_flagged_indicators": int((result.indicator_table["deficiency"] >= 0.75).sum()),
    }
