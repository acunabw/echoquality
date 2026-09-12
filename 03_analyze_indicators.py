from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from echo_quality_pipeline.associations import association_table, bootstrap_stability, select_nonredundant


META = {"patient_id", "view", "quality", "not_apt", "apt_target"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Identifica indicadores significativos, estables y no redundantes")
    parser.add_argument("--input", default="results/patient_view_features.csv")
    parser.add_argument("--output-dir", default="results/indicator_analysis")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--min-frequency", type=float, default=0.65)
    parser.add_argument("--redundancy", type=float, default=0.90)
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    features = [c for c in data.columns if c not in META]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ranking = association_table(data, features)
    ranking.to_csv(out / "association_ranking.csv", index=False)
    stability = bootstrap_stability(data, features, n_bootstrap=args.bootstrap, top_k=args.top_k)
    stability.frequencies.to_csv(out / "feature_stability.csv", index=False)
    stability.selections.to_csv(out / "bootstrap_selections.csv", index=False)
    stable = stability.frequencies[
        (stability.frequencies["selection_frequency"] >= args.min_frequency)
        & (stability.frequencies["sign_consistency"] >= 0.80)
    ]["feature"].tolist()
    if not stable:
        stable = ranking.head(args.top_k)["feature"].tolist()
    selected, rejected = select_nonredundant(data, ranking, stable, threshold=args.redundancy)
    final = ranking[ranking["feature"].isin(selected)].copy()
    final = final.merge(stability.frequencies, on="feature", how="left")
    final.to_csv(out / "stable_nonredundant_indicators.csv", index=False)
    rejected.to_csv(out / "redundancy_rejections.csv", index=False)
    print(f"Indicadores finales: {len(final)}")
    print(final[["feature", "spearman_rho", "selection_frequency"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
