from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityDecision:
    status: str
    final_probability_not_apt: float
    threshold: float
    dominant_level: str | None
    level_probabilities: dict[str, float]
    reason: str


def operational_status(
    probability_not_apt: float,
    threshold: float,
    level_probabilities: dict[str, float],
    *,
    conditional_margin: float = 0.15,
    reference_alarm: bool = False,
) -> QualityDecision:
    probability = float(probability_not_apt)
    dominant_level = max(level_probabilities, key=level_probabilities.get) if level_probabilities else None
    if probability >= threshold:
        return QualityDecision(
            status="NO_APTA",
            final_probability_not_apt=probability,
            threshold=float(threshold),
            dominant_level=dominant_level,
            level_probabilities=dict(level_probabilities),
            reason="final_probability_exceeds_quality_gate_threshold",
        )
    level_alarm = bool(level_probabilities) and max(level_probabilities.values()) >= max(0.5, threshold - conditional_margin)
    if level_alarm or reference_alarm:
        reason = "level_specific_risk" if level_alarm else "good_reference_deviation"
        return QualityDecision(
            status="APTA_CONDICIONADA",
            final_probability_not_apt=probability,
            threshold=float(threshold),
            dominant_level=dominant_level,
            level_probabilities=dict(level_probabilities),
            reason=reason,
        )
    return QualityDecision(
        status="APTA",
        final_probability_not_apt=probability,
        threshold=float(threshold),
        dominant_level=dominant_level,
        level_probabilities=dict(level_probabilities),
        reason="quality_gate_passed_without_level_alarm",
    )
