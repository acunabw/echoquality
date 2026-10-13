"""Image transformations. Processing changes pixels; it does not make decisions."""

from .transforms import TransformationSpec, apply_transformation

__all__ = ["TransformationSpec", "apply_transformation"]
