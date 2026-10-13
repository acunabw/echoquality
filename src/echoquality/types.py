from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImageMetadata:
    path: str | None = None
    original_shape: tuple[int, ...] = ()
    dtype: str = "unknown"
    intensity_min: float = float("nan")
    intensity_max: float = float("nan")
    spacing_yx: tuple[float, float] | None = None
    bit_depth: int | None = None


@dataclass(frozen=True)
class LoadedImage:
    array: np.ndarray
    metadata: ImageMetadata


@dataclass
class HierarchicalFeatures:
    global_features: dict[str, float] = field(default_factory=dict)
    anatomical_features: dict[str, float] = field(default_factory=dict)
    textural_features: dict[str, float] = field(default_factory=dict)

    def by_level(self) -> dict[str, dict[str, float]]:
        return {
            "global": self.global_features,
            "anatomical": self.anatomical_features,
            "textural": self.textural_features,
        }

    def fused(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for block in self.by_level().values():
            overlap = set(out).intersection(block)
            if overlap:
                raise ValueError(f"Duplicate feature names across levels: {sorted(overlap)}")
            out.update(block)
        return out


@dataclass(frozen=True)
class ExperimentPaths:
    root: Path
    manifest: Path
    frame_features: Path
    view_features: Path
    indicator_root: Path
    gate_root: Path
    references: Path
    processing_root: Path
    report_root: Path
    state_file: Path
