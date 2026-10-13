from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnatomicalRegions:
    lv: np.ndarray
    myocardium: np.ndarray
    la: np.ndarray
    foreground: np.ndarray


def regions_from_camus_mask(mask: np.ndarray, shape: tuple[int, int] | None = None) -> AnatomicalRegions:
    labels = np.asarray(mask)
    if shape is not None and labels.shape != shape:
        raise ValueError(f"Mask shape {labels.shape} does not match image shape {shape}")
    lv = labels == 1
    myocardium = labels == 2
    la = labels == 3
    return AnatomicalRegions(lv=lv, myocardium=myocardium, la=la, foreground=labels > 0)


def validate_regions(regions: AnatomicalRegions) -> dict[str, int | bool]:
    overlap = (
        np.logical_and(regions.lv, regions.myocardium).sum()
        + np.logical_and(regions.lv, regions.la).sum()
        + np.logical_and(regions.myocardium, regions.la).sum()
    )
    return {
        "lv_pixels": int(regions.lv.sum()),
        "myocardium_pixels": int(regions.myocardium.sum()),
        "la_pixels": int(regions.la.sum()),
        "foreground_pixels": int(regions.foreground.sum()),
        "overlap_pixels": int(overlap),
        "valid_nonoverlap": bool(overlap == 0),
    }
