from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectiveWeights:
    global_gain: float = 0.35
    anatomical_preservation: float = 0.35
    textural_gain_and_preservation: float = 0.30
    ssim_bonus: float = 0.05
    invalid_penalty: float = 1.0


def hierarchical_objective(
    *,
    global_gain: float,
    anatomical_preservation: float,
    textural_gain_and_preservation: float,
    ssim: float,
    valid: bool,
    weights: ObjectiveWeights,
) -> float:
    value = (
        weights.global_gain * global_gain
        + weights.anatomical_preservation * anatomical_preservation
        + weights.textural_gain_and_preservation * textural_gain_and_preservation
        + weights.ssim_bonus * ssim
    )
    if not valid:
        value -= weights.invalid_penalty
    return float(value)
