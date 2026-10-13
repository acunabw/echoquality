from __future__ import annotations

from itertools import product

from echoquality.processing.global_processing import global_candidates
from echoquality.processing.transforms import TransformationSpec


def textural_candidates() -> list[TransformationSpec]:
    candidates: list[TransformationSpec] = []
    for radius in (1, 2):
        candidates.append(
            TransformationSpec(
                f"median_r{radius}",
                "textural",
                (("median", {"radius": radius}),),
                "Median filtering candidate; retained only if texture and anatomy remain stable.",
            )
        )
    for sigma_color in (0.05, 0.08, 0.12):
        candidates.append(
            TransformationSpec(
                f"bilateral_{sigma_color}",
                "textural",
                (("bilateral", {"sigma_color": sigma_color, "sigma_spatial": 2.0}),),
                "Edge-aware denoising candidate.",
            )
        )
    candidates.append(
        TransformationSpec(
            "anisotropic_diffusion",
            "textural",
            (("anisotropic_diffusion", {"n_iter": 8, "kappa": 25.0, "gamma": 0.10}),),
            "Speckle-oriented edge-preserving candidate; external to Gonzalez & Woods and requires dedicated citation.",
        )
    )
    return candidates


def anatomical_candidates() -> list[TransformationSpec]:
    candidates: list[TransformationSpec] = []
    for radius in (0.8, 1.2):
        for amount in (0.4, 0.8):
            candidates.append(
                TransformationSpec(
                    f"unsharp_r{radius}_a{amount}",
                    "anatomical",
                    (("unsharp", {"radius": radius, "amount": amount}),),
                    "Conservative boundary enhancement; rejected if edge displacement or texture drift is excessive.",
                )
            )
    return candidates


def candidate_transformations(
    processing_families: list[str],
    *,
    max_candidates: int = 48,
) -> list[TransformationSpec]:
    """Condition transformations on the hierarchical diagnosis.

    Severe dropout or anatomical truncation never triggers an enhancement that
    claims to reconstruct missing information.
    """

    identity = TransformationSpec("identity", "none", (), "No processing baseline.")
    if "review_or_reacquisition" in processing_families:
        return [identity]
    blocks: list[list[TransformationSpec]] = []
    if "global_contrast" in processing_families or "global_dynamic_compression_review" in processing_families:
        blocks.append(global_candidates()[1:])
    if "texture_preserving_denoising" in processing_families:
        blocks.append(textural_candidates())
    if "conservative_sharpening" in processing_families or "boundary_enhancement" in processing_families:
        blocks.append(anatomical_candidates())
    candidates: list[TransformationSpec] = [identity]
    for block in blocks:
        candidates.extend(block)
    # Add controlled two-stage combinations only when two distinct levels request them.
    if len(blocks) >= 2:
        for left, right in product(blocks[0][:4], blocks[1][:4]):
            candidates.append(
                TransformationSpec(
                    name=f"{left.name}_then_{right.name}",
                    target_level="multi",
                    steps=left.steps + right.steps,
                    rationale=f"Conditioned combination: {left.rationale} Then: {right.rationale}",
                )
            )
    unique: list[TransformationSpec] = []
    seen: set[str] = set()
    for candidate in candidates:
        label = candidate.label()
        if label not in seen:
            seen.add(label)
            unique.append(candidate)
    return unique[: int(max_candidates)]
