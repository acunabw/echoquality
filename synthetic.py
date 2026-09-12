from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import exposure, filters, morphology


QUALITY_PROBS = ("Good", "Medium", "Poor")


def _ellipse(shape: tuple[int, int], center: tuple[float, float], axes: tuple[float, float], angle: float = 0.0) -> np.ndarray:
    yy, xx = np.indices(shape)
    cy, cx = center
    a, b = axes
    ca, sa = np.cos(angle), np.sin(angle)
    xr = (xx - cx) * ca + (yy - cy) * sa
    yr = -(xx - cx) * sa + (yy - cy) * ca
    return (xr / max(a, 1e-6)) ** 2 + (yr / max(b, 1e-6)) ** 2 <= 1.0


def _fan_mask(size: int, half_angle_deg: float = 38.0) -> np.ndarray:
    yy, xx = np.indices((size, size))
    cx, y0 = (size - 1) / 2.0, 2.0
    dx, dy = xx - cx, yy - y0
    radius = np.sqrt(dx * dx + dy * dy)
    angle = np.degrees(np.arctan2(np.abs(dx), np.maximum(dy, 1e-6)))
    return (dy > 0) & (radius < size * 0.98) & (angle < half_angle_deg)


def generate_echo_image(
    quality: str,
    *,
    view: str = "2CH",
    phase: str = "ED",
    size: int = 160,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera una imagen sintetica tipo B-mode y una mascara 1=LV, 2=miocardio, 3=LA."""

    rng = np.random.default_rng(seed)
    fan = _fan_mask(size)
    yy, xx = np.indices((size, size))
    angle = -0.06 if view == "2CH" else 0.05
    center_x = size * (0.50 + rng.normal(0, 0.015))
    center_y = size * (0.59 + rng.normal(0, 0.012))
    if phase == "ED":
        cavity_axes = (size * 0.16, size * 0.28)
        wall = size * 0.045
    else:
        cavity_axes = (size * 0.12, size * 0.21)
        wall = size * 0.060
    if view == "4CH":
        cavity_axes = (cavity_axes[0] * 1.08, cavity_axes[1] * 0.96)
    # Poor puede simular foreshortening y descentramiento, pero no en todos los casos.
    if quality == "Poor" and rng.random() < 0.60:
        cavity_axes = (cavity_axes[0] * rng.uniform(0.96, 1.05), cavity_axes[1] * rng.uniform(0.76, 0.94))
        center_y += rng.uniform(1, 6)
    cavity = _ellipse((size, size), (center_y, center_x), cavity_axes, angle)
    outer = _ellipse(
        (size, size),
        (center_y, center_x),
        (cavity_axes[0] + wall, cavity_axes[1] + wall),
        angle,
    )
    myo = outer & ~cavity
    la_center = (center_y - cavity_axes[1] * 0.93, center_x + (8 if view == "4CH" else -4))
    la = _ellipse((size, size), la_center, (size * 0.10, size * 0.09), angle * 0.3)
    la &= fan & ~outer

    image = np.zeros((size, size), dtype=float)
    depth = np.clip((yy - 2) / size, 0, 1)
    image[fan] = 0.28 - 0.10 * depth[fan]
    # Textura miocardica sintetica: speckle moderado + patron anisotropico suave.
    fiber = 0.04 * np.sin(0.18 * xx + 0.07 * yy)
    image[myo] = 0.58 + fiber[myo]
    image[cavity] = 0.08
    image[la] = 0.11
    # Bordes brillantes propios del B-mode.
    myo_boundary = morphology.dilation(myo, morphology.disk(1)) ^ morphology.erosion(myo, morphology.disk(1))
    image[myo_boundary & fan] += 0.18
    # Ecos puntuales.
    scatter = rng.random((size, size)) < 0.008
    image[scatter & fan] += rng.uniform(0.15, 0.35, size=int((scatter & fan).sum()))

    if quality == "Good":
        contrast = rng.uniform(0.84, 1.05)
        noise_sd = rng.uniform(0.020, 0.050)
        blur_sigma = rng.uniform(0.35, 0.80)
        dropout = rng.uniform(0.005, 0.045)
        shadow = rng.uniform(0.00, 0.10)
    elif quality == "Medium":
        contrast = rng.uniform(0.60, 0.90)
        noise_sd = rng.uniform(0.040, 0.085)
        blur_sigma = rng.uniform(0.70, 1.55)
        dropout = rng.uniform(0.035, 0.125)
        shadow = rng.uniform(0.08, 0.28)
    elif quality == "Poor":
        contrast = rng.uniform(0.38, 0.76)
        noise_sd = rng.uniform(0.065, 0.135)
        blur_sigma = rng.uniform(1.15, 2.35)
        dropout = rng.uniform(0.095, 0.250)
        shadow = rng.uniform(0.20, 0.55)
    else:
        raise ValueError(quality)

    # Compresion de contraste alrededor de la media del fan.
    mean_fan = float(image[fan].mean())
    image[fan] = mean_fan + contrast * (image[fan] - mean_fan)
    # Speckle multiplicativo y ruido aditivo.
    speckle = rng.lognormal(mean=-0.5 * noise_sd**2, sigma=noise_sd * 2.2, size=image.shape)
    image[fan] *= speckle[fan]
    image[fan] += rng.normal(0.0, noise_sd, size=int(fan.sum()))
    # Dropout principalmente sobre miocardio.
    dropout_map = rng.random(image.shape) < dropout
    dropout_map &= morphology.dilation(myo, morphology.disk(2))
    image[dropout_map] *= rng.uniform(0.02, 0.25)
    # Sombra acustica en forma de banda radial.
    if shadow > 0:
        shadow_center = center_x + rng.uniform(-size * 0.16, size * 0.16)
        width = size * (0.045 if quality == "Medium" else 0.075)
        shadow_mask = fan & (np.abs(xx - shadow_center) < width + 0.15 * np.maximum(yy - 15, 0)) & (yy > size * 0.28)
        image[shadow_mask] *= 1.0 - shadow
    image = filters.gaussian(image, sigma=blur_sigma, preserve_range=True)
    # Ganancia/compresion aleatoria leve.
    gamma = rng.uniform(0.92, 1.08) if quality == "Good" else rng.uniform(0.85, 1.25)
    image = exposure.adjust_gamma(np.clip(image, 0, 1), gamma=gamma)
    image[~fan] = 0.0
    image = np.clip(image, 0.0, 1.0)

    mask = np.zeros((size, size), dtype=np.uint8)
    mask[cavity & fan] = 1
    mask[myo & fan] = 2
    mask[la & fan] = 3
    return image, mask


def create_demo_dataset(
    root: str | Path,
    *,
    n_patients: int = 54,
    views: tuple[str, ...] = ("2CH", "4CH"),
    phases: tuple[str, ...] = ("ED", "ES"),
    size: int = 160,
    seed: int = 2026,
) -> pd.DataFrame:
    """Crea un conjunto sintetico reproducible para ejecutar el demo sin datos clinicos."""

    out = Path(root)
    image_dir = out / "images"
    mask_dir = out / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    # Distribucion deliberadamente cercana a un escenario con clase Poor minoritaria.
    probabilities = np.array([0.52, 0.30, 0.18])
    for patient_index in range(1, n_patients + 1):
        patient_id = f"demo_{patient_index:03d}"
        for view_index, view in enumerate(views):
            quality = str(rng.choice(QUALITY_PROBS, p=probabilities))
            for phase_index, phase in enumerate(phases):
                local_seed = seed + patient_index * 100 + view_index * 10 + phase_index
                image, mask = generate_echo_image(quality, view=view, phase=phase, size=size, seed=local_seed)
                stem = f"{patient_id}_{view}_{phase}"
                image_path = image_dir / f"{stem}.png"
                mask_path = mask_dir / f"{stem}_gt.png"
                Image.fromarray(np.round(image * 255).astype(np.uint8), mode="L").save(image_path)
                Image.fromarray(mask, mode="L").save(mask_path)
                rows.append(
                    {
                        "patient_id": patient_id,
                        "view": view,
                        "phase": phase,
                        "quality": quality,
                        "apt_target": int(quality in {"Good", "Medium"}),
                        "not_apt": int(quality == "Poor"),
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "synthetic": True,
                    }
                )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "manifest.csv", index=False)
    return manifest
