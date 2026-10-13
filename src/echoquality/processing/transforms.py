from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from skimage import exposure, filters, morphology, restoration

from echoquality.data.io import robust_scale


@dataclass(frozen=True)
class TransformationSpec:
    name: str
    target_level: str
    steps: tuple[tuple[str, dict[str, Any]], ...] = field(default_factory=tuple)
    rationale: str = ""

    def label(self) -> str:
        if not self.steps:
            return "identity"
        parts: list[str] = []
        for operation, params in self.steps:
            detail = ",".join(f"{key}={value}" for key, value in sorted(params.items()))
            parts.append(f"{operation}({detail})" if detail else operation)
        return " + ".join(parts)


def anisotropic_diffusion(
    image: np.ndarray,
    *,
    n_iter: int = 8,
    kappa: float = 25.0,
    gamma: float = 0.10,
) -> np.ndarray:
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
        low, high = float(params.get("low", 1.0)), float(params.get("high", 99.0))
        lo, hi = np.percentile(x, [low, high])
        if hi <= lo:
            return x.copy()
        return np.clip(exposure.rescale_intensity(x, in_range=(lo, hi), out_range=(0.0, 1.0)), 0.0, 1.0)
    if operation == "gamma":
        return np.clip(exposure.adjust_gamma(x, gamma=float(params.get("gamma", 1.0))), 0.0, 1.0)
    if operation == "clahe":
        kernel_size = int(params.get("kernel_size", 16))
        clip_limit = float(params.get("clip_limit", 0.01))
        return np.clip(exposure.equalize_adapthist(x, kernel_size=kernel_size, clip_limit=clip_limit), 0.0, 1.0)
    if operation == "median":
        radius = int(params.get("radius", 1))
        return np.clip(filters.median(x, footprint=morphology.disk(radius)), 0.0, 1.0)
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
        return anisotropic_diffusion(
            x,
            n_iter=int(params.get("n_iter", 8)),
            kappa=float(params.get("kappa", 25.0)),
            gamma=float(params.get("gamma", 0.10)),
        )
    if operation == "unsharp":
        return np.clip(
            filters.unsharp_mask(
                x,
                radius=float(params.get("radius", 1.0)),
                amount=float(params.get("amount", 0.6)),
                preserve_range=True,
            ),
            0.0,
            1.0,
        )
    raise ValueError(f"Unknown transformation operation: {operation}")


def apply_transformation(image: np.ndarray, spec: TransformationSpec) -> np.ndarray:
    result = robust_scale(image)
    for operation, params in spec.steps:
        result = apply_operation(result, operation, params)
    return np.clip(result, 0.0, 1.0)
