from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from echo_quality_pipeline.reference import GoodReference


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye el patron multivariado Good")
    parser.add_argument("--input", default="results/patient_view_features.csv")
    parser.add_argument("--indicator-file", default="results/indicator_analysis/stable_nonredundant_indicators.csv")
    parser.add_argument("--ranking", default="results/indicator_analysis/association_ranking.csv")
    parser.add_argument("--output", default="artifacts/good_reference.joblib")
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    indicators = pd.read_csv(args.indicator_file)["feature"].astype(str).tolist()
    ranking = pd.read_csv(args.ranking) if Path(args.ranking).exists() else None
    reference = GoodReference.fit(data, indicators, association_ranking=ranking)
    reference.save(args.output)
    print(f"Referencia Good guardada con {reference.n_good} observaciones y {len(indicators)} indicadores")


if __name__ == "__main__":
    main()
