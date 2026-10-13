from __future__ import annotations

from echoquality.processing.transforms import TransformationSpec


def global_candidates() -> list[TransformationSpec]:
    candidates = [
        TransformationSpec("identity", "global", (), "No processing baseline."),
        TransformationSpec(
            "percentile_stretch",
            "global",
            (("percentile_stretch", {"low": 1.0, "high": 99.0}),),
            "Robust contrast expansion; does not represent physical dynamic-range recovery.",
        ),
    ]
    for gamma in (0.75, 0.90, 1.10, 1.25):
        candidates.append(
            TransformationSpec(
                f"gamma_{gamma}",
                "global",
                (("gamma", {"gamma": gamma}),),
                "Point intensity transformation for dark/bright imbalance.",
            )
        )
    for clip in (0.008, 0.015, 0.025):
        for kernel in (8, 16, 24):
            candidates.append(
                TransformationSpec(
                    f"clahe_{clip}_{kernel}",
                    "global",
                    (("clahe", {"clip_limit": clip, "kernel_size": kernel}),),
                    "Local histogram processing candidate; must pass texture-preservation constraints.",
                )
            )
    return candidates
