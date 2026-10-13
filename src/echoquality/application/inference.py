from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from echoquality.analysis.extractor import HierarchicalFeatureExtractor
from echoquality.config import load_config
from echoquality.constants import LEVELS, feature_level
from echoquality.data.aggregation import aggregate_patient_view
from echoquality.data.io import load_image_with_metadata, load_mask, save_grayscale, save_mask
from echoquality.optimization.filter_optimizer import AdaptiveOptimizationResult, optimize_adaptive_processing
from echoquality.optimization.objective import ObjectiveWeights
from echoquality.recognition.diagnostics import diagnose_one
from echoquality.recognition.good_reference import GoodReferenceBundle, load_references
from echoquality.recognition.hierarchical_gate import load_gate
from echoquality.recognition.fusion import HierarchicalGateBundle, predict_late_fusion
from echoquality.types import ImageMetadata


@dataclass
class CaseFrame:
    phase: str
    image: np.ndarray
    mask: np.ndarray | None
    metadata: ImageMetadata
    image_name: str = ""
    mask_name: str = ""


@dataclass
class CaseInput:
    patient_id: str
    view: str
    frames: dict[str, CaseFrame]
    source: str = "manual"


@dataclass
class InferenceRuntime:
    experiment_root: Path
    config_path: Path
    config: dict[str, Any]
    extractor: HierarchicalFeatureExtractor
    gate: HierarchicalGateBundle
    view_references: GoodReferenceBundle
    frame_references: GoodReferenceBundle


@dataclass
class CaseAnalysisResult:
    mode: str
    patient_id: str
    view: str
    summary: dict[str, Any]
    frame_features: pd.DataFrame
    view_features: pd.DataFrame
    prediction: pd.DataFrame
    level_summary: pd.DataFrame
    deviations: pd.DataFrame
    processing: dict[str, AdaptiveOptimizationResult] = field(default_factory=dict)
    source_frames: dict[str, CaseFrame] = field(default_factory=dict)


def _artifact_paths(experiment_root: str | Path) -> dict[str, Path]:
    root = Path(experiment_root)
    return {
        "gate": root / "quality_gate" / "hierarchical_quality_gate.joblib",
        "view_reference": root / "references" / "good_reference_view.joblib",
        "frame_reference": root / "references" / "good_reference_frame.joblib",
        "resolved_config": root / "reports" / "resolved_config.yaml",
        "manifest": root / "manifest.csv",
    }


def runtime_status(experiment_root: str | Path, config_path: str | Path | None = None) -> pd.DataFrame:
    paths = _artifact_paths(experiment_root)
    rows = []
    for name in ("gate", "view_reference", "frame_reference", "manifest"):
        path = paths[name]
        rows.append({"resource": name, "path": str(path), "exists": path.exists()})
    chosen = Path(config_path) if config_path else paths["resolved_config"]
    rows.append({"resource": "config", "path": str(chosen), "exists": chosen.exists()})
    return pd.DataFrame(rows)


def load_runtime(
    experiment_root: str | Path,
    *,
    config_path: str | Path | None = None,
) -> InferenceRuntime:
    root = Path(experiment_root)
    paths = _artifact_paths(root)
    selected_config = Path(config_path) if config_path else paths["resolved_config"]
    missing = [path for path in (paths["gate"], paths["view_reference"], paths["frame_reference"], selected_config) if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "The localhost requires a completed experiment. Missing resources:\n" + formatted
        )
    config = load_config(selected_config)
    return InferenceRuntime(
        experiment_root=root,
        config_path=selected_config,
        config=config,
        extractor=HierarchicalFeatureExtractor.from_config(config),
        gate=load_gate(paths["gate"]),
        view_references=load_references(paths["view_reference"]),
        frame_references=load_references(paths["frame_reference"]),
    )


def _load_frame(image_path: str | Path, mask_path: str | Path | None, phase: str) -> CaseFrame:
    loaded = load_image_with_metadata(image_path)
    mask = load_mask(mask_path) if mask_path else None
    if mask is not None and mask.shape != loaded.array.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {loaded.array.shape} for {phase}")
    return CaseFrame(
        phase=str(phase).upper(),
        image=loaded.array,
        mask=mask,
        metadata=loaded.metadata,
        image_name=Path(image_path).name,
        mask_name=Path(mask_path).name if mask_path else "",
    )


def load_case_from_manifest(
    manifest_path: str | Path,
    *,
    patient_id: str,
    view: str,
    replacement_phase: str | None = None,
    replacement_image_path: str | Path | None = None,
    replacement_mask_path: str | Path | None = None,
) -> CaseInput:
    manifest = pd.read_csv(manifest_path)
    required = {"patient_id", "view", "phase", "image_path", "mask_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    subset = manifest[
        (manifest["patient_id"].astype(str) == str(patient_id))
        & (manifest["view"].astype(str).str.upper() == str(view).upper())
    ].copy()
    if subset.empty:
        raise ValueError(f"No manifest rows for patient={patient_id}, view={view}")
    frames: dict[str, CaseFrame] = {}
    replacement_phase_upper = str(replacement_phase).upper() if replacement_phase else None
    for phase in ("ED", "ES"):
        rows = subset[subset["phase"].astype(str).str.upper() == phase]
        if rows.empty:
            raise ValueError(f"The complete Quality Gate requires both ED and ES; missing {phase}")
        row = rows.iloc[0]
        image_path = row["image_path"]
        mask_path = row["mask_path"]
        if replacement_phase_upper == phase and replacement_image_path:
            image_path = replacement_image_path
            if replacement_mask_path:
                mask_path = replacement_mask_path
        frames[phase] = _load_frame(image_path, mask_path, phase)
    return CaseInput(patient_id=str(patient_id), view=str(view).upper(), frames=frames, source="manifest")


def _extract_tables(runtime: InferenceRuntime, case: CaseInput) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for phase in ("ED", "ES"):
        if phase not in case.frames:
            raise ValueError("The complete Quality Gate requires ED and ES frames")
        frame = case.frames[phase]
        hierarchy = runtime.extractor.extract(frame.image, frame.mask, metadata=frame.metadata)
        row: dict[str, Any] = {
            "patient_id": case.patient_id,
            "view": case.view,
            "phase": phase,
            "quality": "Unknown",
            "apt_target": np.nan,
            "not_apt": np.nan,
            "image_path": frame.image_name,
            "mask_path": frame.mask_name,
            "image_height": int(frame.image.shape[0]),
            "image_width": int(frame.image.shape[1]),
            "original_dtype": frame.metadata.dtype,
            "original_bit_depth": frame.metadata.bit_depth,
            "spacing_y_mm": frame.metadata.spacing_yx[0] if frame.metadata.spacing_yx else None,
            "spacing_x_mm": frame.metadata.spacing_yx[1] if frame.metadata.spacing_yx else None,
        }
        row.update(hierarchy.fused())
        rows.append(row)
    frame_table = pd.DataFrame(rows)
    feature_columns = [column for column in frame_table.columns if feature_level(str(column))]
    modes = tuple(runtime.config.get("features", {}).get("temporal_aggregation", ["mean", "abs_delta"]))
    view_table = aggregate_patient_view(frame_table, feature_columns, modes=modes)
    view_table["quality"] = "Unknown"
    view_table["not_apt"] = np.nan
    view_table["apt_target"] = np.nan
    return frame_table, view_table


def _processing_settings(runtime: InferenceRuntime) -> tuple[dict[str, Any], ObjectiveWeights]:
    settings = runtime.config.get("processing", {})
    weight_config = settings.get("objective_weights", {})
    weights = ObjectiveWeights(
        global_gain=float(weight_config.get("global_gain", 0.35)),
        anatomical_preservation=float(weight_config.get("anatomical_preservation", 0.35)),
        textural_gain_and_preservation=float(weight_config.get("textural_gain_and_preservation", 0.30)),
        ssim_bonus=float(weight_config.get("ssim_bonus", 0.05)),
        invalid_penalty=float(weight_config.get("invalid_penalty", 1.0)),
    )
    return settings, weights


def analyze_complete_case(
    runtime: InferenceRuntime,
    case: CaseInput,
    *,
    run_processing: bool = True,
    force_processing: bool = False,
) -> CaseAnalysisResult:
    frame_table, view_table = _extract_tables(runtime, case)
    required_features = {
        feature
        for model in runtime.gate.level_models.values()
        for feature in model.feature_names
    }
    missing = sorted(required_features - set(view_table.columns))
    if missing:
        raise ValueError(
            "The case aggregation does not match the trained model. Missing features: "
            + ", ".join(missing[:12])
        )
    prediction = predict_late_fusion(runtime.gate, view_table)
    settings = runtime.config.get("recognition", {}).get("diagnostics", {})
    diagnosis = diagnose_one(
        view_table.iloc[0],
        prediction.iloc[0],
        runtime.view_references,
        runtime.gate,
        top_n=int(settings.get("top_n", 5)),
        conditional_margin=float(settings.get("conditional_margin", 0.15)),
    )
    level_rows: list[dict[str, Any]] = []
    deviation_rows: list[dict[str, Any]] = []
    for level in LEVELS:
        level_diag = diagnosis.levels[level]
        level_rows.append(
            {
                "level": level,
                "probability_not_apt": level_diag.probability_not_apt,
                "adequacy_score": 1.0 - level_diag.probability_not_apt,
                "mahalanobis_distance": level_diag.mahalanobis_distance,
                "reference_threshold_q95": level_diag.reference_threshold_q95,
                "reference_alarm": level_diag.reference_alarm,
                "issues": ";".join(level_diag.issue_categories),
            }
        )
        for item in level_diag.top_deviations:
            deviation_rows.append({"level": level, **item})
    processing: dict[str, AdaptiveOptimizationResult] = {}
    processing_settings, weights = _processing_settings(runtime)
    eligible = diagnosis.decision.status in set(processing_settings.get("apply_to_statuses", ["APTA_CONDICIONADA"]))
    can_process = all(case.frames[phase].mask is not None for phase in ("ED", "ES"))
    if run_processing and can_process and diagnosis.processing_families and (eligible or force_processing):
        constraints = processing_settings.get("constraints", {})
        for phase in ("ED", "ES"):
            frame = case.frames[phase]
            processing[phase] = optimize_adaptive_processing(
                frame.image,
                frame.mask,
                metadata=frame.metadata,
                references=runtime.frame_references,
                reference_stratum=f"{case.view}_{phase}",
                processing_families=diagnosis.processing_families,
                extractor=runtime.extractor,
                max_candidates=int(processing_settings.get("max_candidates", 48)),
                ssim_min=float(constraints.get("ssim_min", 0.82)),
                edge_f1_min=float(constraints.get("edge_f1_min", 0.70)),
                anatomical_drift_max=float(constraints.get("anatomical_drift_max", 0.15)),
                texture_drift_max=float(constraints.get("texture_drift_max", 0.20)),
                clipping_fraction_max=float(constraints.get("clipping_fraction_max", 0.20)),
                weights=weights,
                min_objective_improvement=float(processing_settings.get("min_objective_improvement", 0.02)),
            )
    summary = {
        "patient_id": case.patient_id,
        "view": case.view,
        "source": case.source,
        "status": diagnosis.decision.status,
        "probability_not_apt": diagnosis.decision.final_probability_not_apt,
        "threshold": diagnosis.decision.threshold,
        "decision_reason": diagnosis.decision.reason,
        "dominant_level": diagnosis.decision.dominant_level,
        "dominant_issue": diagnosis.dominant_issue,
        "processing_families": diagnosis.processing_families,
        "score_global": diagnosis.decision.level_probabilities.get("global"),
        "score_anatomical": diagnosis.decision.level_probabilities.get("anatomical"),
        "score_textural": diagnosis.decision.level_probabilities.get("textural"),
        "processing_executed": bool(processing),
    }
    return CaseAnalysisResult(
        mode="complete",
        patient_id=case.patient_id,
        view=case.view,
        summary=summary,
        frame_features=frame_table,
        view_features=view_table,
        prediction=prediction,
        level_summary=pd.DataFrame(level_rows),
        deviations=pd.DataFrame(deviation_rows),
        processing=processing,
        source_frames=case.frames,
    )


def explore_single_frame(
    runtime: InferenceRuntime,
    frame: CaseFrame,
    *,
    view: str,
    run_processing: bool = False,
) -> CaseAnalysisResult:
    hierarchy = runtime.extractor.extract(frame.image, frame.mask, metadata=frame.metadata)
    fused = hierarchy.fused()
    row = {
        "patient_id": "uploaded_single_frame",
        "view": str(view).upper(),
        "phase": frame.phase,
        "quality": "Unknown",
        **fused,
    }
    frame_table = pd.DataFrame([row])
    level_rows: list[dict[str, Any]] = []
    deviations: list[dict[str, Any]] = []
    issue_map: dict[str, str] = {}
    for level in LEVELS:
        reference = runtime.frame_references.get(level, f"{str(view).upper()}_{frame.phase}")
        z = reference.robust_z(row)
        ordered = sorted(z.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
        distance = reference.mahalanobis(row)
        alarm = bool(distance > reference.distance_q95)
        level_rows.append(
            {
                "level": level,
                "mahalanobis_distance": distance,
                "reference_threshold_q95": reference.distance_q95,
                "reference_alarm": alarm,
            }
        )
        for feature, value in ordered:
            deviations.append(
                {
                    "level": level,
                    "feature": feature,
                    "robust_z": float(value),
                    "direction": "above_good" if value > 0 else "below_good",
                }
            )
            issue_map[feature] = level
    summary = {
        "status": "EXPLORATORY_SINGLE_FRAME",
        "message": "A single frame can be compared with frame-level Good References, but the trained patient-view Quality Gate requires ED and ES.",
        "view": str(view).upper(),
        "phase": frame.phase,
        "processing_executed": False,
    }
    return CaseAnalysisResult(
        mode="single_frame",
        patient_id="uploaded_single_frame",
        view=str(view).upper(),
        summary=summary,
        frame_features=frame_table,
        view_features=pd.DataFrame(),
        prediction=pd.DataFrame(),
        level_summary=pd.DataFrame(level_rows),
        deviations=pd.DataFrame(deviations),
        processing={},
        source_frames={frame.phase: frame},
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def save_case_analysis(result: CaseAnalysisResult, output_root: str | Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_patient = "".join(char if char.isalnum() or char in "-_" else "_" for char in result.patient_id)
    case_root = Path(output_root) / f"{timestamp}_{safe_patient}_{result.view}"
    case_root.mkdir(parents=True, exist_ok=True)
    (case_root / "summary.json").write_text(
        json.dumps(result.summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    result.frame_features.to_csv(case_root / "frame_features.csv", index=False)
    if not result.view_features.empty:
        result.view_features.to_csv(case_root / "view_features.csv", index=False)
    if not result.prediction.empty:
        result.prediction.to_csv(case_root / "prediction.csv", index=False)
    result.level_summary.to_csv(case_root / "level_summary.csv", index=False)
    result.deviations.to_csv(case_root / "deviations.csv", index=False)
    for phase, frame in result.source_frames.items():
        save_grayscale(case_root / f"{phase}_original.png", frame.image)
        if frame.mask is not None:
            save_mask(case_root / f"{phase}_mask.png", frame.mask)
    for phase, optimization in result.processing.items():
        phase_root = case_root / f"processing_{phase}"
        phase_root.mkdir(parents=True, exist_ok=True)
        optimization.candidates.to_csv(phase_root / "candidate_scores.csv", index=False)
        save_grayscale(phase_root / "recommended.png", optimization.best_image)
        (phase_root / "recommendation.json").write_text(
            json.dumps(
                {
                    "recommendation": optimization.recommendation,
                    "transformation": optimization.best_spec.label(),
                    "target_level": optimization.best_spec.target_level,
                    "rationale": optimization.best_spec.rationale,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return case_root
