from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from skimage import exposure, filters, morphology, restoration

from .image_io import robust_scale


@dataclass(frozen=True)
class TransformationSpec:
    name: str
    steps: tuple[tuple[str, dict[str, Any]], ...] = field(default_factory=tuple)

    def label(self) -> str:
        if not self.steps:
            return "original_sin_cambios"
        parts = []
        for operation, params in self.steps:
            detail = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
            parts.append(f"{operation}({detail})" if detail else operation)
        return " + ".join(parts)


def _anisotropic_diffusion(image: np.ndarray, n_iter: int = 8, kappa: float = 25.0, gamma: float = 0.12) -> np.ndarray:
    """Difusion anisotropica de Perona-Malik, implementacion ligera 2D."""

    u = np.asarray(image, dtype=float).copy()
    kappa01 = max(kappa / 255.0, 1e-4)
    for _ in range(int(n_iter)):
        north = np.roll(u, -1, axis=0) - u
        south = np.roll(u, 1, axis=0) - u
        east = np.roll(u, -1, axis=1) - u
        west = np.roll(u, 1, axis=1) - u
        c_n = np.exp(-(north / kappa01) ** 2)
        c_s = np.exp(-(south / kappa01) ** 2)
        c_e = np.exp(-(east / kappa01) ** 2)
        c_w = np.exp(-(west / kappa01) ** 2)
        u += gamma * (c_n * north + c_s * south + c_e * east + c_w * west)
    return np.clip(u, 0.0, 1.0)


def apply_operation(image01: np.ndarray, operation: str, params: dict[str, Any]) -> np.ndarray:
    x = np.asarray(image01, dtype=float)
    if operation == "percentile_stretch":
        low = float(params.get("low", 1.0))
        high = float(params.get("high", 99.0))
        lo, hi = np.percentile(x, [low, high])
        return np.clip(exposure.rescale_intensity(x, in_range=(lo, hi), out_range=(0.0, 1.0)), 0.0, 1.0)
    if operation == "gamma":
        return np.clip(exposure.adjust_gamma(x, gamma=float(params.get("gamma", 1.0))), 0.0, 1.0)
    if operation == "clahe":
        kernel_size = int(params.get("kernel_size", 16))
        clip_limit = float(params.get("clip_limit", 0.01))
        return np.clip(exposure.equalize_adapthist(x, kernel_size=kernel_size, clip_limit=clip_limit), 0.0, 1.0)
    if operation == "median":
        radius = int(params.get("radius", 1))
        return filters.median(x, footprint=morphology.disk(radius))
    if operation == "gaussian":
        return np.clip(filters.gaussian(x, sigma=float(params.get("sigma", 1.0)), preserve_range=True), 0.0, 1.0)
    if operation == "bilateral":
        return np.clip(
            restoration.denoise_bilateral(
                x,
                sigma_color=float(params.get("sigma_color", 0.08)),
                sigma_spatial=float(params.get("sigma_spatial", 2.0)),
                channel_axis=None,
            ),
            0.0,
            1.0,
        )
    if operation == "anisotropic_diffusion":
        return _anisotropic_diffusion(
            x,
            n_iter=int(params.get("n_iter", 8)),
            kappa=float(params.get("kappa", 25.0)),
            gamma=float(params.get("gamma", 0.12)),
        )
    if operation == "unsharp":
        return np.clip(
            filters.unsharp_mask(
                x,
                radius=float(params.get("radius", 1.0)),
                amount=float(params.get("amount", 0.8)),
                preserve_range=True,
            ),
            0.0,
            1.0,
        )
    raise ValueError(f"Operacion desconocida: {operation}")


def apply_transformation(image: np.ndarray, spec: TransformationSpec) -> np.ndarray:
    out = robust_scale(image)
    for operation, params in spec.steps:
        out = apply_operation(out, operation, params)
    return np.clip(out, 0.0, 1.0)


def candidate_transformations(top_issues: list[str], max_candidates: int = 48) -> list[TransformationSpec]:
    """Construye una rejilla pequeña y explicable, guiada por el diagnostico."""

    candidates: list[TransformationSpec] = [TransformationSpec("identity", ())]
    if "contrast" in top_issues or "brightness_dynamic_range" in top_issues:
        for clip in (0.008, 0.015, 0.025):
            for kernel in (8, 16, 24):
                candidates.append(TransformationSpec("clahe", (("clahe", {"clip_limit": clip, "kernel_size": kernel}),)))
        for gamma in (0.75, 0.90, 1.10, 1.25):
            candidates.append(TransformationSpec("gamma", (("gamma", {"gamma": gamma}),)))
        candidates.append(TransformationSpec("stretch", (("percentile_stretch", {"low": 1.0, "high": 99.0}),)))
    if "noise_speckle" in top_issues:
        for radius in (1, 2, 3):
            candidates.append(TransformationSpec("median", (("median", {"radius": radius}),)))
        for sigma_color in (0.05, 0.08, 0.12):
            candidates.append(
                TransformationSpec(
                    "bilateral",
                    (("bilateral", {"sigma_color": sigma_color, "sigma_spatial": 2.0}),),
                )
            )
        candidates.append(
            TransformationSpec(
                "anisotropic",
                (("anisotropic_diffusion", {"n_iter": 8, "kappa": 25.0, "gamma": 0.10}),),
            )
        )
    if "sharpness" in top_issues:
        for radius in (0.8, 1.2, 1.8):
            for amount in (0.4, 0.8, 1.2):
                candidates.append(TransformationSpec("unsharp", (("unsharp", {"radius": radius, "amount": amount}),)))
    # Combinaciones controladas para los casos mas comunes.
    if "noise_speckle" in top_issues and "contrast" in top_issues:
        for radius in (1, 2):
            for clip in (0.008, 0.015):
                candidates.append(
                    TransformationSpec(
                        "denoise_then_clahe",
                        (
                            ("median", {"radius": radius}),
                            ("clahe", {"clip_limit": clip, "kernel_size": 16}),
                        ),
                    )
                )
    if "contrast" in top_issues and "sharpness" in top_issues:
        for clip in (0.008, 0.015):
            candidates.append(
                TransformationSpec(
                    "clahe_then_unsharp",
                    (
                        ("clahe", {"clip_limit": clip, "kernel_size": 16}),
                        ("unsharp", {"radius": 1.0, "amount": 0.6}),
                    ),
                )
            )
    # Elimina duplicados preservando orden.
    seen: set[str] = set()
    unique: list[TransformationSpec] = []
    for spec in candidates:
        label = spec.label()
        if label not in seen:
            seen.add(label)
            unique.append(spec)
    return unique[:max_candidates]
