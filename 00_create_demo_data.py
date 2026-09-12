from __future__ import annotations

import argparse
from pathlib import Path

from echo_quality_pipeline.synthetic import create_demo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Crea imagenes ecocardiograficas sinteticas para el demo")
    parser.add_argument("--output", default="data/demo")
    parser.add_argument("--patients", type=int, default=54)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    manifest = create_demo_dataset(Path(args.output), n_patients=args.patients, seed=args.seed)
    print(f"Demo creado: {len(manifest)} imagenes; {manifest['patient_id'].nunique()} pacientes")
    print(manifest.groupby('quality').size())


if __name__ == "__main__":
    main()
