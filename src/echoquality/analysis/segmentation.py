from __future__ import annotations

from typing import Protocol

import numpy as np


class Segmenter(Protocol):
    """Interface for a validated segmentation model used by downstream studies."""

    def predict(self, image: np.ndarray) -> np.ndarray:
        ...


class ProvidedMaskSegmenter:
    """Adapter used when CAMUS ground-truth masks are supplied in the manifest."""

    def __init__(self, mask: np.ndarray) -> None:
        self._mask = np.asarray(mask)

    def predict(self, image: np.ndarray) -> np.ndarray:
        if self._mask.shape != np.asarray(image).shape:
            raise ValueError("Provided mask and image shapes differ")
        return self._mask.copy()
