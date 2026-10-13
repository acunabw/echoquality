from __future__ import annotations

import argparse
from pathlib import Path

from echoquality.config import load_config
from echoquality.experiments.orchestrator import STAGE_ORDER, run_experiment
from echoquality.reporting.reports import compare_experiment_summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="echoquality",
        description="EchoQuality 2.0 hierarchical pattern-recognition pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run a configured experiment")
    run.add_argument("--config", required=True, help="Experiment YAML file")
    run.add_argument(
        "--stages",
        nargs="+",
        default=["all"],
        choices=["all", *STAGE_ORDER],
        help="Stages to execute; dependencies are added automatically",
    )
    run.add_argument("--no-resume", action="store_true", help="Do not skip completed stages")
    run.add_argument("--force", action="store_true", help="Re-run selected stages")
    run.add_argument("--fast", action="store_true", help="Use reduced CV grids for engineering tests")

    compare = subparsers.add_parser("compare", help="Compare experiment_summary.json files")
    compare.add_argument("summaries", nargs="+", help="Summary JSON paths")
    compare.add_argument("--output", default="results/experiment_comparison.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        config = load_config(args.config)
        run_experiment(
            config,
            stages=args.stages,
            resume=not args.no_resume,
            force=args.force,
            fast=True if args.fast else None,
        )
        return 0
    if args.command == "compare":
        table = compare_experiment_summaries(args.summaries)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output, index=False)
        print(table.to_string(index=False))
        print(f"Saved: {output}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
