from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from echo_quality_pipeline.features import extract_features
from echo_quality_pipeline.image_io import load_image, load_mask, save_grayscale
from echo_quality_pipeline.optimization import optimize_image
from echo_quality_pipeline.quality_gate import load_bundle, predict_quality_gate
from echo_quality_pipeline.reference import GoodReference
from echo_quality_pipeline.reporting import save_before_after, save_candidate_scores, save_deviation_plot


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostica indicadores y optimiza filtros/parametros")
    parser.add_argument("--manifest", default="data/demo/manifest.csv")
    parser.add_argument("--row", type=int, default=-1, help="Fila del manifiesto; -1 selecciona una Medium")
    parser.add_argument("--quality-model", default="artifacts/quality_gate.joblib")
    parser.add_argument("--reference", default="artifacts/good_reference.joblib")
    parser.add_argument("--output-dir", default="results/stage2_processing")
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    if args.row >= 0:
        row = manifest.iloc[args.row]
    else:
        medium = manifest[manifest["quality"] == "Medium"]
        row = medium.iloc[0] if len(medium) else manifest.iloc[0]
    image = load_image(row["image_path"])
    mask = load_mask(row["mask_path"]) if str(row.get("mask_path", "")) else None
    features = extract_features(image, mask)
    gate = predict_quality_gate(load_bundle(args.quality_model), features)
    reference = GoodReference.load(args.reference)
    result = optimize_image(image, mask, reference, gate_status=gate["status"])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_grayscale(out / "original.png", image)
    save_grayscale(out / "recommended.png", result.best_image)
    result.candidates.to_csv(out / "candidate_transformations.csv", index=False)
    result.diagnostic.indicator_table.to_csv(out / "indicator_deviations.csv", index=False)
    result.diagnostic.issue_table.to_csv(out / "issue_summary.csv", index=False)
    save_deviation_plot(result.diagnostic.indicator_table, out / "indicator_deviations.png")
    save_candidate_scores(result.candidates, out / "candidate_scores.png")
    save_before_after(image, result.best_image, out / "before_after.png", result.recommendation)
    summary = {
        "patient_id": row.get("patient_id"),
        "view": row.get("view"),
        "phase": row.get("phase"),
        "reference_quality": row.get("quality"),
        "gate": gate,
        "top_issues": result.diagnostic.top_issues,
        "recommendation": result.recommendation,
        "best_transformation": result.best_spec.label(),
    }
    (out / "optimization_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
