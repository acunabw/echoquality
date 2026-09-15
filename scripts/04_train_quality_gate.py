from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from echo_quality_pipeline.quality_gate import fit_final_quality_gate, nested_group_validation, save_bundle
from echo_quality_pipeline.reporting import save_confusion, save_roc


META = {"patient_id", "view", "quality", "not_apt", "apt_target"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena la compuerta APTA/NO APTA con validacion por paciente")
    parser.add_argument("--input", default="results/patient_view_features.csv")
    parser.add_argument("--indicator-file", default="results/indicator_analysis/stable_nonredundant_indicators.csv")
    parser.add_argument("--output-dir", default="results/stage1_quality_gate")
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--outer", type=int, default=5)
    parser.add_argument("--inner", type=int, default=4)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    if Path(args.indicator_file).exists():
        indicators = pd.read_csv(args.indicator_file)["feature"].astype(str).tolist()
    else:
        indicators = [c for c in data.columns if c not in META]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    models = ("LogisticRegression",) if args.fast else ("LogisticRegression", "RandomForest")
    cv = nested_group_validation(
        data,
        indicators,
        outer_splits=3 if args.fast else args.outer,
        inner_splits=2 if args.fast else args.inner,
        model_names=models,
    )
    cv.fold_metrics.to_csv(out / "fold_metrics.csv", index=False)
    cv.predictions.to_csv(out / "oof_predictions.csv", index=False)
    cv.summary.to_csv(out / "model_summary.csv", index=False)
    cv.selected_features.to_csv(out / "selected_features_by_fold.csv", index=False)
    best_model = str(cv.summary.iloc[0]["model"])
    bundle = fit_final_quality_gate(
        data,
        indicators,
        model_name=best_model,
        inner_splits=2 if args.fast else args.inner,
    )
    artifact_path = save_bundle(bundle, Path(args.artifacts) / "quality_gate.joblib")
    save_roc(cv.predictions, out / "roc_quality_gate.png", model_name=best_model)
    save_confusion(cv.predictions, out / "confusion_quality_gate.png", model_name=best_model)
    print(cv.summary.to_string(index=False))
    print(f"Modelo final: {artifact_path}")


if __name__ == "__main__":
    main()
