from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from echoquality.data.legacy_v1 import convert_v1_view_features
from echoquality.recognition.hierarchical_gate import fit_final_gate, nested_group_validation, save_gate


def main() -> None:
    parser = argparse.ArgumentParser(description="Architectural ablation using legacy v1 measurements")
    parser.add_argument("--input", required=True, help="v1 view_features.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["LogisticRegression", "RandomForest"],
        choices=["LogisticRegression", "RandomForest"],
        help="Model families for this architectural ablation.",
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    converted = convert_v1_view_features(pd.read_csv(args.input))
    converted.to_csv(output / "legacy_features_prefixed.csv", index=False)
    result = nested_group_validation(
        converted,
        model_names=tuple(args.models),
        outer_splits=3 if args.fast else 5,
        inner_splits=2 if args.fast else 4,
        fast=args.fast,
    )
    result.fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    result.predictions.to_csv(output / "oof_predictions.csv", index=False)
    result.summary.to_csv(output / "model_summary.csv", index=False)
    result.selected_features.to_csv(output / "selected_features_by_fold.csv", index=False)
    best = str(result.summary.iloc[0]["model"])
    bundle = fit_final_gate(converted, model_name=best, inner_splits=2 if args.fast else 4, fast=args.fast)
    save_gate(bundle, output / "hierarchical_gate_legacy_features.joblib")
    print(result.summary.to_string(index=False))
    print("WARNING: this is an architectural ablation on v1 measurements, not a full v2 feature re-extraction.")


if __name__ == "__main__":
    main()
