from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_REQUIRED = {"patient_id", "view", "phase", "quality", "image_path"}


def validate_manifest(manifest: pd.DataFrame, *, require_masks: bool = True) -> dict[str, object]:
    missing = _REQUIRED - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    quality_values = set(manifest["quality"].astype(str).unique())
    invalid_quality = quality_values - {"Good", "Medium", "Poor"}
    if invalid_quality:
        raise ValueError(f"Unexpected quality labels: {sorted(invalid_quality)}")
    duplicate_keys = manifest.duplicated(["patient_id", "view", "phase"]).sum()
    missing_images = sum(not Path(str(path)).exists() for path in manifest["image_path"])
    missing_masks = 0
    if require_masks:
        if "mask_path" not in manifest:
            raise ValueError("Manifest has no mask_path column")
        missing_masks = sum(not str(path).strip() or not Path(str(path)).exists() for path in manifest["mask_path"])
    return {
        "n_rows": int(len(manifest)),
        "n_patients": int(manifest["patient_id"].nunique()),
        "duplicate_patient_view_phase": int(duplicate_keys),
        "missing_images": int(missing_images),
        "missing_masks": int(missing_masks),
        "quality_counts": manifest["quality"].value_counts().to_dict(),
    }


def nested_patient_order(manifest: pd.DataFrame, *, seed: int = 2026) -> list[str]:
    """Return a reproducibly shuffled patient order for nested learning curves."""

    patients = np.asarray(sorted(manifest["patient_id"].astype(str).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    return patients.tolist()


def select_patient_subset(
    manifest: pd.DataFrame,
    n_patients: int | None,
    *,
    seed: int = 2026,
    order_file: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Select a reproducible nested subset without relying on dataset file order."""

    all_patients = set(manifest["patient_id"].astype(str).unique())
    if order_file and Path(order_file).exists():
        raw_order = [
            line.strip()
            for line in Path(order_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        order = [patient for patient in raw_order if patient in all_patients]
        if raw_order and not order:
            raise ValueError(
                f"Patient order file {order_file} does not match this dataset. "
                "Use a dataset-specific order file or delete it so it can be regenerated."
            )
        missing = np.asarray(sorted(all_patients - set(order)))
        if missing.size:
            # Preserve the existing prefix while adding newly available patients
            # in a deterministic random order rather than source-file order.
            rng = np.random.default_rng(seed)
            rng.shuffle(missing)
            order.extend(missing.tolist())
    else:
        order = nested_patient_order(manifest, seed=seed)
        if order_file:
            output = Path(order_file)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("\n".join(order) + "\n", encoding="utf-8")
    count = len(order) if n_patients in (None, 0) else min(int(n_patients), len(order))
    selected = order[:count]
    subset = manifest[manifest["patient_id"].astype(str).isin(selected)].copy()
    subset["patient_selection_rank"] = subset["patient_id"].astype(str).map({patient: index for index, patient in enumerate(order)})
    subset = subset.sort_values(["patient_selection_rank", "view", "phase"]).reset_index(drop=True)
    return subset, selected
