from __future__ import annotations

LEVELS: tuple[str, ...] = ("global", "anatomical", "textural")
QUALITY_ORDER: dict[str, int] = {"Poor": 0, "Medium": 1, "Good": 2}
QUALITY_TO_TARGET: dict[str, int] = {"Poor": 1, "Medium": 0, "Good": 0}

IDENTIFIER_COLUMNS: set[str] = {
    "patient_id",
    "view",
    "phase",
    "image_path",
    "mask_path",
    "cfg_path",
    "source_dataset",
    "synthetic",
}
TARGET_COLUMNS: set[str] = {"quality", "apt_target", "not_apt"}
META_COLUMNS: set[str] = IDENTIFIER_COLUMNS | TARGET_COLUMNS

AGGREGATION_SUFFIXES: tuple[str, ...] = (
    "__mean",
    "__abs_delta",
    "__ed",
    "__es",
    "__min",
    "__max",
)


def base_feature_name(column: str) -> str:
    """Remove a temporal aggregation suffix from a feature column."""

    for suffix in AGGREGATION_SUFFIXES:
        if column.endswith(suffix):
            return column[: -len(suffix)]
    return column


def feature_level(column: str) -> str | None:
    """Infer the representation level from a prefixed feature name."""

    base = base_feature_name(column)
    for level in LEVELS:
        if base.startswith(f"{level}__"):
            return level
    return None
