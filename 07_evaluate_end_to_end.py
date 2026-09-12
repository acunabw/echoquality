from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from echo_quality_pipeline.features import extract_features
from echo_quality_pipeline.image_io import load_image, load_mask
from echo_quality_pipeline.optimization import optimize_image
from echo_quality_pipeline.quality_gate import load_bundle, predict_quality_gate
from echo_quality_pipeline.reference import GoodReference


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalua el pipeline completo sobre un subconjunto")
    parser.add_argument("--manifest", default="data/demo/manifest.csv")
    parser.add_argument("--quality-model", default="artifacts/quality_gate.joblib")
    parser.add_argument("--reference", default="artifacts/good_reference.joblib")
    parser.add_argument("--output-dir", default="results/end_to_end")
    parser.add_argument("--max-images", type=int, default=10)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    # Una fase por paciente-vista para no contar dos veces la misma etiqueta.
    manifest = manifest.sort_values(["patient_id", "view", "phase"]).drop_duplicates(["patient_id", "view"])
    # Se priorizan Medium para demostrar la segunda etapa y se agregan algunos Good/Poor.
    sample = pd.concat(
        [
            manifest[manifest["quality"] == "Medium"].head(max(1, args.max_images // 2)),
            manifest[manifest["quality"] == "Good"].head(max(1, args.max_images // 3)),
            manifest[manifest["quality"] == "Poor"].head(max(1, args.max_images // 6)),
        ],
        ignore_index=True,
    ).head(args.max_images)
    model = load_bundle(args.quality_model)
    reference = GoodReference.load(args.reference)
    rows = []
    for _, item in sample.iterrows():
        image = load_image(item["image_path"])
        mask = load_mask(item["mask_path"]) if str(item.get("mask_path", "")) else None
        features = extract_features(image, mask)
        gate = predict_quality_gate(model, features)
        result = optimize_image(image, mask, reference, gate_status=gate["status"], max_candidates=10)
        best_row = result.candidates.iloc[0] if len(result.candidates) else pd.Series(dtype=float)
        rows.append(
            {
                "patient_id": item["patient_id"],
                "view": item["view"],
                "quality_reference": item["quality"],
                "gate_status": gate["status"],
                "proba_not_apt": gate["proba_not_apt"],
                "top_issues": ";".join(result.diagnostic.top_issues),
                "recommendation": result.recommendation,
                "best_transformation": result.best_spec.label(),
                "quality_gain": float(best_row.get("quality_gain", 0.0)),
                "contrast_gain": float(best_row.get("contrast_gain", 0.0)),
                "noise_reduction": float(best_row.get("noise_reduction", 0.0)),
                "dice_gain_proxy": float(best_row.get("dice_gain_proxy", 0.0)),
                "ssim": float(best_row.get("ssim_original", 1.0)),
                "texture_drift": float(best_row.get("texture_drift", 0.0)),
            }
        )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "end_to_end_results.csv", index=False)
    summary = {
        "n_images": int(len(table)),
        "gate_status_counts": table["gate_status"].value_counts().to_dict(),
        "mean_quality_gain_medium": float(table.loc[table["quality_reference"] == "Medium", "quality_gain"].mean()),
        "mean_contrast_gain_medium": float(table.loc[table["quality_reference"] == "Medium", "contrast_gain"].mean()),
        "mean_noise_reduction_medium": float(table.loc[table["quality_reference"] == "Medium", "noise_reduction"].mean()),
        "mean_dice_gain_proxy_medium": float(table.loc[table["quality_reference"] == "Medium", "dice_gain_proxy"].mean()),
        "mean_ssim_processed": float(table["ssim"].mean()),
        "mean_texture_drift": float(table["texture_drift"].mean()),
        "note": "Resultados ilustrativos obtenidos con datos sinteticos; no representan desempeno clinico ni CAMUS real.",
    }
    (out / "end_to_end_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
