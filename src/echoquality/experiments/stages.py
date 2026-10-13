from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from echoquality.analysis.extractor import HierarchicalFeatureExtractor
from echoquality.config import save_resolved_config
from echoquality.constants import LEVELS, feature_level
from echoquality.data.aggregation import aggregate_patient_view
from echoquality.data.camus import build_manifest
from echoquality.data.io import load_image_with_metadata, load_mask, save_grayscale
from echoquality.data.manifests import select_patient_subset, validate_manifest
from echoquality.data.synthetic import create_demo_dataset
from echoquality.experiments.audit import audit_feature_tables, audit_manifest
from echoquality.experiments.paths import build_experiment_paths
from echoquality.optimization.filter_optimizer import optimize_adaptive_processing
from echoquality.optimization.objective import ObjectiveWeights
from echoquality.recognition.associations import discover_indicators
from echoquality.recognition.diagnostics import diagnose_dataset
from echoquality.recognition.good_reference import (
    build_good_references,
    load_references,
    save_references,
)
from echoquality.recognition.hierarchical_gate import (
    fit_final_gate,
    load_gate,
    nested_group_validation,
    save_gate,
)
from echoquality.reporting.plots import save_confusion, save_level_score_boxplot, save_roc
from echoquality.reporting.reports import experiment_summary, save_json


def _timer() -> float:
    return time.perf_counter()


def _elapsed(start: float) -> float:
    return float(time.perf_counter() - start)


def stage_manifest(config: dict[str, Any]) -> dict[str, Any]:
    start = _timer()
    paths = build_experiment_paths(config)
    paths.root.mkdir(parents=True, exist_ok=True)
    dataset = config.get("dataset", {})
    kind = str(dataset.get("kind", "camus")).lower()
    if kind == "synthetic":
        demo_root = Path(dataset.get("synthetic_root", f"data/synthetic/{config['experiment']['name']}"))
        master = create_demo_dataset(
            demo_root,
            n_patients=int(dataset.get("available_patients", dataset.get("n_patients", 54))),
            seed=int(config.get("experiment", {}).get("seed", 2026)),
            size=int(dataset.get("image_size", 160)),
        )
    elif kind == "camus":
        master_path = Path(dataset.get("master_manifest", "data/camus_manifest_all.csv"))
        if master_path.exists():
            master = pd.read_csv(master_path)
        else:
            camus_root = str(dataset.get("camus_root", "")).strip()
            if not camus_root or "${" in camus_root:
                raise ValueError(
                    "CAMUS root is unresolved. Set the CAMUS_ROOT environment variable "
                    "or provide dataset.camus_root in the experiment YAML before building "
                    "a master manifest."
                )
            master = build_manifest(camus_root)
            master_path.parent.mkdir(parents=True, exist_ok=True)
            master.to_csv(master_path, index=False)
    else:
        raise ValueError(f"Unsupported dataset kind: {kind}")
        # Normalizar etiquetas operacionales
    if "quality" not in master.columns:
        raise ValueError(
            "El manifest no contiene la columna quality"
        )

    quality = master["quality"].astype("string").str.strip()

    mapping = {
        "Good": 1,
        "Medium": 1,
        "Poor": 0,
    }

    apt = quality.map(mapping)

    if apt.isna().any():
        invalid = quality[apt.isna()].unique().tolist()
        raise ValueError(
            f"Etiquetas de calidad desconocidas: {invalid}"
        )

    master["quality"] = quality
    master["apt_target"] = apt.astype(int)
    master["not_apt"] = 1 - master["apt_target"]
    
    order_file = Path(dataset.get("patient_order_file", "data/patient_order_seed2026.txt"))
    subset, selected = select_patient_subset(
        master,
        dataset.get("n_patients"),
        seed=int(config.get("experiment", {}).get("seed", 2026)),
        order_file=order_file,
    )
    paths.manifest.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(paths.manifest, index=False)
    audit = audit_manifest(subset)
    audit_path = paths.report_root / "manifest_audit.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.checks.to_csv(audit_path, index=False)
    if not audit.passed:
        raise RuntimeError(f"Manifest audit failed; inspect {audit_path}")
    return {
        "manifest": str(paths.manifest),
        "n_patients": len(selected),
        "n_frames": len(subset),
        "quality_counts": subset["quality"].value_counts().to_dict(),
        "elapsed_seconds": _elapsed(start),
    }


def stage_features(config: dict[str, Any]) -> dict[str, Any]:
    start = _timer()
    paths = build_experiment_paths(config)
    manifest = pd.read_csv(paths.manifest)
    feature_config = config.get("features", {})
    extractor = HierarchicalFeatureExtractor.from_config(config)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, source in manifest.iterrows():
        try:
            loaded = load_image_with_metadata(str(source["image_path"]))
            mask_path = str(source.get("mask_path", "")).strip()
            mask = load_mask(mask_path) if mask_path else None
            hierarchy = extractor.extract(loaded.array, mask, metadata=loaded.metadata)
            row = source.to_dict()
            row.update(
                {
                    "image_height": int(loaded.array.shape[0]),
                    "image_width": int(loaded.array.shape[1]),
                    "original_dtype": loaded.metadata.dtype,
                    "original_bit_depth": loaded.metadata.bit_depth,
                    "spacing_y_mm": loaded.metadata.spacing_yx[0] if loaded.metadata.spacing_yx else None,
                    "spacing_x_mm": loaded.metadata.spacing_yx[1] if loaded.metadata.spacing_yx else None,
                }
            )
            row.update(hierarchy.fused())
            rows.append(row)
        except Exception as exc:  # keep a complete, auditable failure table
            failures.append({"row": int(index), "image_path": source.get("image_path"), "error": repr(exc)})
            if bool(feature_config.get("fail_on_error", True)):
                raise
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No feature rows were produced")
    feature_columns = [column for column in frame.columns if feature_level(str(column))]
    view = aggregate_patient_view(
        frame,
        feature_columns,
        modes=tuple(feature_config.get("temporal_aggregation", ["mean", "abs_delta"])),
    )
    paths.frame_features.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(paths.frame_features, index=False)
    view.to_csv(paths.view_features, index=False)
    extractor.catalog().save_csv(paths.frame_features.parent / "feature_catalog.csv")
    if failures:
        pd.DataFrame(failures).to_csv(paths.frame_features.parent / "extraction_failures.csv", index=False)
    audit = audit_feature_tables(frame, view)
    paths.report_root.mkdir(parents=True, exist_ok=True)
    audit.checks.to_csv(paths.report_root / "feature_audit.csv", index=False)
    if not audit.passed:
        raise RuntimeError(f"Feature audit failed; inspect {paths.report_root / 'feature_audit.csv'}")
    return {
        "frame_features": str(paths.frame_features),
        "view_features": str(paths.view_features),
        "frame_shape": list(frame.shape),
        "view_shape": list(view.shape),
        "n_failures": len(failures),
        "elapsed_seconds": _elapsed(start),
    }


def stage_indicators(config: dict[str, Any]) -> dict[str, Any]:
    start = _timer()
    paths = build_experiment_paths(config)
    data = pd.read_csv(paths.view_features)
    settings = config.get("indicator_discovery", {})
    summary: dict[str, Any] = {}
    for level in LEVELS:
        columns = [column for column in data.columns if feature_level(str(column)) == level]
        outputs = discover_indicators(
            data,
            columns,
            bootstrap=int(settings.get("n_bootstrap", 200)),
            top_k=int(settings.get("top_k", 15)),
            min_frequency=float(settings.get("min_selection_frequency", 0.65)),
            min_sign_consistency=float(settings.get("min_sign_consistency", 0.80)),
            redundancy=float(settings.get("redundancy_abs_spearman", 0.90)),
            seed=int(config.get("experiment", {}).get("seed", 2026)),
        )
        level_root = paths.indicator_root / level
        level_root.mkdir(parents=True, exist_ok=True)
        for name, table in outputs.items():
            table.to_csv(level_root / f"{name}.csv", index=False)
        final = outputs["stable_nonredundant_indicators"]
        summary[level] = {
            "n_candidates": len(columns),
            "n_exploratory_indicators": len(final),
            "fallback_used": bool(final["fallback_used"].any()) if not final.empty else False,
            "warning": "Exploratory list is not supplied to nested CV.",
        }
    save_json(summary, paths.indicator_root / "indicator_summary.json")
    summary["elapsed_seconds"] = _elapsed(start)
    return summary


def stage_quality_gate(config: dict[str, Any], *, fast_override: bool | None = None) -> dict[str, Any]:
    start = _timer()
    paths = build_experiment_paths(config)
    data = pd.read_csv(paths.view_features)
    settings = config.get("recognition", {}).get("quality_gate", {})
    fast = bool(settings.get("fast", False) if fast_override is None else fast_override)
    model_names = tuple(settings.get("models", ["LogisticRegression", "RandomForest"]))
    validation = nested_group_validation(
        data,
        model_names=model_names,
        outer_splits=int(settings.get("outer_splits", 5)),
        inner_splits=int(settings.get("inner_splits", 4)),
        seed=int(config.get("experiment", {}).get("seed", 2026)),
        fast=fast,
        min_sensitivity_not_apt=float(settings.get("min_sensitivity_not_apt", 0.90)),
    )
    paths.gate_root.mkdir(parents=True, exist_ok=True)
    validation.fold_metrics.to_csv(paths.gate_root / "fold_metrics.csv", index=False)
    validation.predictions.to_csv(paths.gate_root / "oof_predictions.csv", index=False)
    validation.summary.to_csv(paths.gate_root / "model_summary.csv", index=False)
    validation.selected_features.to_csv(paths.gate_root / "selected_features_by_fold.csv", index=False)
    best_model = str(validation.summary.iloc[0]["model"])
    final = fit_final_gate(
        data,
        model_name=best_model,
        inner_splits=int(settings.get("inner_splits", 4)),
        seed=int(config.get("experiment", {}).get("seed", 2026)),
        fast=fast,
        min_sensitivity_not_apt=float(settings.get("min_sensitivity_not_apt", 0.90)),
    )
    artifact = save_gate(final, paths.gate_root / "hierarchical_quality_gate.joblib")
    save_roc(validation.predictions, paths.gate_root / "roc_quality_gate.png", model_name=best_model)
    save_confusion(validation.predictions, paths.gate_root / "confusion_quality_gate.png", model_name=best_model)
    save_level_score_boxplot(validation.predictions, paths.gate_root / "level_scores.png", model_name=best_model)
    return {
        "best_model": best_model,
        "artifact": str(artifact),
        "threshold": final.threshold,
        "threshold_rule": final.threshold_rule,
        "best_metrics": validation.summary.iloc[0].to_dict(),
        "elapsed_seconds": _elapsed(start),
    }


def stage_references(config: dict[str, Any]) -> dict[str, Any]:
    start = _timer()
    paths = build_experiment_paths(config)
    view = pd.read_csv(paths.view_features)
    frame = pd.read_csv(paths.frame_features)
    view_bundle = build_good_references(view, stratify_by_view=True)
    frame = frame.copy()
    frame["_reference_stratum"] = frame["view"].astype(str) + "_" + frame["phase"].astype(str)
    frame_bundle = build_good_references(frame, view_column="_reference_stratum", stratify_by_view=True)
    paths.references.mkdir(parents=True, exist_ok=True)
    view_path = save_references(view_bundle, paths.references / "good_reference_view.joblib")
    frame_path = save_references(frame_bundle, paths.references / "good_reference_frame.joblib")
    return {
        "view_reference": str(view_path),
        "frame_reference": str(frame_path),
        "n_good_views": view_bundle.metadata.get("n_good"),
        "n_good_frames": frame_bundle.metadata.get("n_good"),
        "elapsed_seconds": _elapsed(start),
    }


def stage_diagnostics(config: dict[str, Any]) -> dict[str, Any]:
    start = _timer()
    paths = build_experiment_paths(config)
    data = pd.read_csv(paths.view_features)
    gate = load_gate(paths.gate_root / "hierarchical_quality_gate.joblib")
    references = load_references(paths.references / "good_reference_view.joblib")
    settings = config.get("recognition", {}).get("diagnostics", {})
    summary, deviations = diagnose_dataset(
        data,
        gate,
        references,
        top_n=int(settings.get("top_n", 5)),
        conditional_margin=float(settings.get("conditional_margin", 0.15)),
    )
    output = paths.root / "diagnostics"
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "diagnostic_summary.csv", index=False)
    deviations.to_csv(output / "indicator_deviations.csv", index=False)
    return {
        "diagnostic_summary": str(output / "diagnostic_summary.csv"),
        "status_counts": summary["status"].value_counts().to_dict(),
        "dominant_level_counts": summary["dominant_level"].value_counts().to_dict(),
        "elapsed_seconds": _elapsed(start),
    }


def stage_adaptive_processing(config: dict[str, Any]) -> dict[str, Any]:
    start = _timer()
    paths = build_experiment_paths(config)
    settings = config.get("processing", {})
    if not bool(settings.get("enabled", True)):
        return {"skipped": True, "reason": "processing.enabled=false", "elapsed_seconds": _elapsed(start)}
    diagnostics_path = paths.root / "diagnostics" / "diagnostic_summary.csv"
    diagnostics = pd.read_csv(diagnostics_path)
    frame = pd.read_csv(paths.frame_features)
    references = load_references(paths.references / "good_reference_frame.joblib")
    extractor = HierarchicalFeatureExtractor.from_config(config)
    eligible_statuses = set(settings.get("apply_to_statuses", ["APTA_CONDICIONADA"]))
    eligible = diagnostics[diagnostics["status"].isin(eligible_statuses)].copy()
    eligible = eligible.sort_values("probability_not_apt", ascending=False).head(int(settings.get("max_cases", 12)))
    constraints = settings.get("constraints", {})
    weight_config = settings.get("objective_weights", {})
    weights = ObjectiveWeights(
        global_gain=float(weight_config.get("global_gain", 0.35)),
        anatomical_preservation=float(weight_config.get("anatomical_preservation", 0.35)),
        textural_gain_and_preservation=float(weight_config.get("textural_gain_and_preservation", 0.30)),
        ssim_bonus=float(weight_config.get("ssim_bonus", 0.05)),
        invalid_penalty=float(weight_config.get("invalid_penalty", 1.0)),
    )
    summaries: list[dict[str, Any]] = []
    for _, diagnosis in eligible.iterrows():
        patient_id, view = str(diagnosis["patient_id"]), str(diagnosis["view"])
        families = [value for value in str(diagnosis.get("processing_families", "")).split(";") if value]
        cases = frame[(frame["patient_id"].astype(str) == patient_id) & (frame["view"].astype(str) == view)]
        for _, case in cases.iterrows():
            loaded = load_image_with_metadata(str(case["image_path"]))
            mask = load_mask(str(case["mask_path"]))
            phase = str(case["phase"])
            result = optimize_adaptive_processing(
                loaded.array,
                mask,
                metadata=loaded.metadata,
                references=references,
                reference_stratum=f"{view}_{phase}",
                processing_families=families,
                extractor=extractor,
                max_candidates=int(settings.get("max_candidates", 48)),
                ssim_min=float(constraints.get("ssim_min", 0.82)),
                edge_f1_min=float(constraints.get("edge_f1_min", 0.70)),
                anatomical_drift_max=float(constraints.get("anatomical_drift_max", 0.15)),
                texture_drift_max=float(constraints.get("texture_drift_max", 0.20)),
                clipping_fraction_max=float(constraints.get("clipping_fraction_max", 0.20)),
                weights=weights,
                min_objective_improvement=float(settings.get("min_objective_improvement", 0.02)),
            )
            case_root = paths.processing_root / f"{patient_id}_{view}_{phase}"
            case_root.mkdir(parents=True, exist_ok=True)
            result.candidates.to_csv(case_root / "candidate_scores.csv", index=False)
            save_grayscale(case_root / "original.png", loaded.array)
            save_grayscale(case_root / "recommended.png", result.best_image)
            save_json(
                {
                    "patient_id": patient_id,
                    "view": view,
                    "phase": phase,
                    "recommendation": result.recommendation,
                    "best_transformation": result.best_spec.label(),
                    "target_level": result.best_spec.target_level,
                    "processing_families": families,
                },
                case_root / "optimization_summary.json",
            )
            summaries.append(
                {
                    "patient_id": patient_id,
                    "view": view,
                    "phase": phase,
                    "recommendation": result.recommendation,
                    "best_transformation": result.best_spec.label(),
                    "case_root": str(case_root),
                }
            )
    paths.processing_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(paths.processing_root / "processing_summary.csv", index=False)
    return {
        "n_processed_frames": len(summaries),
        "processing_summary": str(paths.processing_root / "processing_summary.csv"),
        "elapsed_seconds": _elapsed(start),
    }


def stage_report(config: dict[str, Any]) -> dict[str, Any]:
    start = _timer()
    paths = build_experiment_paths(config)
    manifest = pd.read_csv(paths.manifest)
    view = pd.read_csv(paths.view_features)
    model_summary = pd.read_csv(paths.gate_root / "model_summary.csv")
    indicator_tables: dict[str, pd.DataFrame] = {}
    for level in LEVELS:
        path = paths.indicator_root / level / "stable_nonredundant_indicators.csv"
        indicator_tables[level] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    summary = experiment_summary(
        experiment_name=str(config["experiment"]["name"]),
        manifest=manifest,
        view_features=view,
        model_summary=model_summary,
        indicator_tables=indicator_tables,
    )
    paths.report_root.mkdir(parents=True, exist_ok=True)
    save_json(summary, paths.report_root / "experiment_summary.json")
    save_resolved_config(config, paths.report_root / "resolved_config.yaml")
    markdown = [
        f"# Experiment report: {summary['experiment']}",
        "",
        f"- Patients: {summary['n_patients']}",
        f"- Frames: {summary['n_frames']}",
        f"- Patient-views: {summary['n_patient_views']}",
        f"- Best model: {summary['best_model'].get('model', 'NA')}",
        f"- ROC AUC: {summary['best_model'].get('roc_auc_mean', 'NA')}",
        f"- Balanced accuracy: {summary['best_model'].get('balanced_accuracy_mean', 'NA')}",
        f"- Sensitivity NO APTA: {summary['best_model'].get('sensitivity_not_apt_mean', 'NA')}",
        f"- Specificity APTA: {summary['best_model'].get('specificity_apt_mean', 'NA')}",
        "",
        "The exploratory indicator lists are reported for interpretation only; supervised feature selection used for performance estimation is fitted inside nested patient-group cross-validation.",
    ]
    (paths.report_root / "experiment_report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return {"summary": str(paths.report_root / "experiment_summary.json"), "elapsed_seconds": _elapsed(start)}


STAGE_FUNCTIONS = {
    "manifest": stage_manifest,
    "features": stage_features,
    "indicators": stage_indicators,
    "gate": stage_quality_gate,
    "references": stage_references,
    "diagnostics": stage_diagnostics,
    "processing": stage_adaptive_processing,
    "report": stage_report,
}
