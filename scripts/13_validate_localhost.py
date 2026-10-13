from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from echoquality.application import analyze_complete_case, load_case_from_manifest, load_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate EchoQuality 2.0 localhost artifacts and one complete case")
    parser.add_argument("--experiment-root", default="results/smoke")
    parser.add_argument("--config", default="configs/experiments/smoke.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--patient", default=None)
    parser.add_argument("--view", default=None)
    parser.add_argument("--processing", action="store_true")
    args = parser.parse_args()

    root = Path(args.experiment_root)
    runtime = load_runtime(root, config_path=args.config)
    manifest_path = Path(args.manifest) if args.manifest else root / "manifest.csv"
    manifest = pd.read_csv(manifest_path)
    patient = str(args.patient) if args.patient else str(manifest.iloc[0]["patient_id"])
    patient_rows = manifest[manifest["patient_id"].astype(str) == patient]
    view = str(args.view) if args.view else str(patient_rows.iloc[0]["view"])
    case = load_case_from_manifest(manifest_path, patient_id=patient, view=view)
    result = analyze_complete_case(runtime, case, run_processing=args.processing)
    print(json.dumps(result.summary, indent=2, ensure_ascii=False, default=str))
    print("level_rows=", len(result.level_summary))
    print("deviation_rows=", len(result.deviations))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
