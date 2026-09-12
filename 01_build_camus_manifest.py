from __future__ import annotations

import argparse
from pathlib import Path

from echo_quality_pipeline.camus import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Construye el manifiesto CAMUS con Good/Medium/Poor")
    parser.add_argument("--camus-root", required=True)
    parser.add_argument("--output", default="data/camus_manifest.csv")
    args = parser.parse_args()
    manifest = build_manifest(args.camus_root)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    print(f"Manifiesto guardado en {out}")
    print(manifest.groupby(["view", "quality"]).size())


if __name__ == "__main__":
    main()
