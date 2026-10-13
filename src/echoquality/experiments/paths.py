from __future__ import annotations

from pathlib import Path
from typing import Any

from echoquality.types import ExperimentPaths


def build_experiment_paths(config: dict[str, Any]) -> ExperimentPaths:
    name = str(config["experiment"]["name"])
    root = Path(config.get("output", {}).get("root", f"results/{name}"))
    manifest = Path(config.get("dataset", {}).get("experiment_manifest", root / "manifest.csv"))
    return ExperimentPaths(
        root=root,
        manifest=manifest,
        frame_features=root / "features" / "frame_features.csv",
        view_features=root / "features" / "view_features.csv",
        indicator_root=root / "indicator_analysis",
        gate_root=root / "quality_gate",
        references=root / "references",
        processing_root=root / "adaptive_processing",
        report_root=root / "reports",
        state_file=root / "experiment_state.json",
    )
