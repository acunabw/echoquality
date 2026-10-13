from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def audit(root: Path, output: Path) -> None:
    rows = []
    for n in (10, 50, 100, 250, 500):
        experiment = root / "results" / f"camus{n}"
        if not experiment.exists():
            continue
        frame = pd.read_csv(experiment / "frame_features.csv")
        view = pd.read_csv(experiment / "view_features.csv")
        summary = pd.read_csv(experiment / "stage1_quality_gate" / "model_summary.csv")
        indicators = pd.read_csv(experiment / "indicator_analysis" / "stable_nonredundant_indicators.csv")
        best = summary.iloc[0]
        stable = int(
            (
                (indicators.get("selection_frequency", pd.Series(dtype=float)) >= 0.65)
                & (indicators.get("sign_consistency", pd.Series(dtype=float)) >= 0.80)
            ).sum()
        )
        quality_counts = view["quality"].value_counts()
        rows.append(
            {
                "experiment": f"CAMUS{n}",
                "patients": int(view["patient_id"].nunique()),
                "frames": len(frame),
                "patient_views": len(view),
                "good": int(quality_counts.get("Good", 0)),
                "medium": int(quality_counts.get("Medium", 0)),
                "poor": int(quality_counts.get("Poor", 0)),
                "reported_indicators": len(indicators),
                "indicators_meeting_config_threshold": stable,
                "silent_fallback_likely": bool(stable == 0 and len(indicators) > 0),
                "best_model": best["model"],
                "roc_auc": best["roc_auc_mean"],
                "balanced_accuracy": best["balanced_accuracy_mean"],
                "sensitivity_not_apt": best["sensitivity_not_apt_mean"],
                "specificity_apt": best["specificity_apt_mean"],
                "brier": best["brier_mean"],
            }
        )
    table = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "v1_experiment_comparison.csv", index=False)
    if (root / "results" / "camus500" / "frame_features.csv").exists():
        frame = pd.read_csv(root / "results" / "camus500" / "frame_features.csv")
        view = pd.read_csv(root / "results" / "camus500" / "view_features.csv")
        metadata = {
            "patient_id", "view", "phase", "quality", "apt_target", "not_apt",
            "image_path", "mask_path", "cfg_path",
        }
        common_numeric = [
            column
            for column in frame.columns
            if column in view.columns
            and column not in metadata
            and pd.api.types.is_numeric_dtype(frame[column])
            and pd.api.types.is_numeric_dtype(view[column])
        ]
        expected_view = frame.groupby(["patient_id", "view"], as_index=False)[common_numeric].mean()
        observed_view = view[["patient_id", "view", *common_numeric]]
        comparison = expected_view.merge(
            observed_view, on=["patient_id", "view"], suffixes=("_expected", "_observed"), validate="one_to_one"
        )
        view_error = max(
            float(
                np.nanmax(
                    np.abs(
                        comparison[f"{column}_expected"]
                        - comparison[f"{column}_observed"]
                    )
                )
            )
            for column in common_numeric
        )
        identities = {
            "view_features_max_abs_error_vs_ED_ES_mean": view_error,
            "dynamic_range_identity_max_abs_error": float(
                np.nanmax(np.abs(frame["dynamic_range_p95_p05"] - (frame["intensity_p95"] - frame["intensity_p05"])))
            ),
            "speckle_identity_max_abs_error": float(
                np.nanmax(
                    np.abs(
                        frame["speckle_index_global"]
                        - frame["intensity_std"] / frame["intensity_mean"].abs().clip(lower=1e-12)
                    )
                )
            ),
            "mask_fraction_identity_max_abs_error": float(
                np.nanmax(
                    np.abs(
                        frame["mask_foreground_fraction"]
                        - (frame["lv_fraction"] + frame["myo_fraction"] + frame["la_fraction"])
                    )
                )
            ),
            "glcm_energy_squared_asm_max_abs_error": float(
                np.nanmax(np.abs(frame["glcm_energy"] ** 2 - frame["glcm_asm"]))
            ),
        }
        (output / "v1_identity_audit.json").write_text(json.dumps(identities, indent=2), encoding="utf-8")
    print(table.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit an EchoQuality 1.x repository and its CAMUS results")
    parser.add_argument("--v1-root", required=True)
    parser.add_argument("--output", default="docs/v1_audit_generated")
    args = parser.parse_args()
    audit(Path(args.v1_root), Path(args.output))


if __name__ == "__main__":
    main()
