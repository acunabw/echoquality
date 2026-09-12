from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from echo_quality_pipeline.evaluation import aggregate_patient_view
from echo_quality_pipeline.features import extract_features
from echo_quality_pipeline.image_io import load_image, load_mask


META_COLUMNS = ["patient_id", "view", "phase", "quality", "apt_target", "not_apt", "image_path", "mask_path", "synthetic"]


def extract_table(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i, row in manifest.iterrows():
        image = load_image(row["image_path"])
        mask_path = str(row.get("mask_path", "")).strip()
        mask = load_mask(mask_path) if mask_path and Path(mask_path).exists() else None
        features = extract_features(image, mask)
        metadata = {column: row[column] for column in META_COLUMNS if column in manifest.columns}
        rows.append({**metadata, **features})
        if (i + 1) % 50 == 0:
            print(f"Procesadas {i + 1}/{len(manifest)} imagenes")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae indicadores de intensidad, textura y morfologia")
    parser.add_argument("--manifest", default="data/camus_manifest.csv")
    parser.add_argument("--frame-output", default="results/frame_features.csv")
    parser.add_argument("--view-output", default="results/patient_view_features.csv")
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest)
    if "not_apt" not in manifest:
        manifest["not_apt"] = (manifest["quality"] == "Poor").astype(int)
    if "apt_target" not in manifest:
        manifest["apt_target"] = 1 - manifest["not_apt"]
    frame = extract_table(manifest)
    frame_path = Path(args.frame_output)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(frame_path, index=False)
    feature_columns = [c for c in frame.columns if c not in META_COLUMNS]
    view = aggregate_patient_view(frame, feature_columns)
    view_path = Path(args.view_output)
    view.to_csv(view_path, index=False)
    print(f"Caracteristicas por cuadro: {frame_path}")
    print(f"Caracteristicas agregadas por paciente-vista: {view_path}")


if __name__ == "__main__":
    main()
