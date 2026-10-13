from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage import exposure, filters

QUALITY_VALUES = ("Good", "Medium", "Poor")


def generate_echo_image(
    quality: str,
    *,
    view: str = "2CH",
    phase: str = "ED",
    size: int = 160,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a small ultrasound-like phantom for engineering smoke tests only."""

    if quality not in QUALITY_VALUES:
        raise ValueError(quality)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    center_x = size * (0.50 + rng.normal(0, 0.015))
    center_y = size * (0.56 + rng.normal(0, 0.015))
    scale = 1.0 if phase.upper() == "ED" else 0.82
    rx = size * (0.16 if view == "2CH" else 0.18) * scale
    ry = size * 0.25 * scale
    cavity = ((xx - center_x) / rx) ** 2 + ((yy - center_y) / ry) ** 2 <= 1.0
    outer = ((xx - center_x) / (rx * 1.23)) ** 2 + ((yy - center_y) / (ry * 1.20)) ** 2 <= 1.0
    myo = outer & ~cavity
    la = ((xx - center_x) / (rx * 0.86)) ** 2 + ((yy - (center_y - ry * 1.08)) / (ry * 0.48)) ** 2 <= 1.0
    fan = (yy > size * 0.05) & (np.abs(xx - size / 2) < (yy + size * 0.10) * 0.68)

    image = np.zeros((size, size), dtype=float)
    base = 0.10 + 0.10 * rng.random((size, size))
    speckle = rng.rayleigh(scale=0.14, size=(size, size))
    image[fan] = base[fan] + speckle[fan]
    image[cavity & fan] *= 0.34
    image[myo & fan] += 0.32 + 0.20 * rng.random(np.count_nonzero(myo & fan))
    image[la & fan] *= 0.42

    if quality == "Good":
        blur_sigma, noise_sigma, dropout, gamma = 0.55, 0.018, 0.00, rng.uniform(0.95, 1.05)
    elif quality == "Medium":
        blur_sigma, noise_sigma, dropout, gamma = 1.15, 0.045, 0.20, rng.uniform(0.86, 1.18)
    else:
        blur_sigma, noise_sigma, dropout, gamma = 2.00, 0.075, 0.48, rng.uniform(0.74, 1.35)
    image += rng.normal(0, noise_sigma, image.shape)
    if dropout:
        shadow_center = center_x + rng.uniform(-size * 0.12, size * 0.12)
        width = size * (0.04 if quality == "Medium" else 0.075)
        shadow = fan & (np.abs(xx - shadow_center) < width + 0.11 * np.maximum(yy - size * 0.15, 0)) & (yy > size * 0.30)
        image[shadow] *= 1.0 - dropout
    image = filters.gaussian(image, sigma=blur_sigma, preserve_range=True)
    image = exposure.adjust_gamma(np.clip(image, 0, 1), gamma=gamma)
    image[~fan] = 0.0

    mask = np.zeros((size, size), dtype=np.uint8)
    mask[cavity & fan] = 1
    mask[myo & fan] = 2
    mask[la & fan] = 3
    return np.clip(image, 0.0, 1.0), mask


def create_demo_dataset(
    root: str | Path,
    *,
    n_patients: int = 54,
    seed: int = 2026,
    size: int = 160,
) -> pd.DataFrame:
    output = Path(root)
    image_dir = output / "images"
    mask_dir = output / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for patient_index in range(1, n_patients + 1):
        patient_id = f"demo_{patient_index:03d}"
        for view_index, view in enumerate(("2CH", "4CH")):
            quality = str(rng.choice(QUALITY_VALUES, p=(0.52, 0.30, 0.18)))
            for phase_index, phase in enumerate(("ED", "ES")):
                image, mask = generate_echo_image(
                    quality,
                    view=view,
                    phase=phase,
                    size=size,
                    seed=seed + patient_index * 100 + view_index * 10 + phase_index,
                )
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
                        "image_path": str(image_path.resolve()),
                        "mask_path": str(mask_path.resolve()),
                        "source_dataset": "SYNTHETIC_DEMO",
                        "synthetic": True,
                    }
                )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(output / "manifest.csv", index=False)
    return manifest
