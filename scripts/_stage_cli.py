from __future__ import annotations

import argparse

from echoquality.config import load_config
from echoquality.experiments.orchestrator import run_experiment


def run_stage(stage: str) -> None:
    parser = argparse.ArgumentParser(description=f"Run EchoQuality 2.0 stage: {stage}")
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    run_experiment(
        config,
        stages=[stage],
        resume=not args.force,
        force=args.force,
        fast=True if args.fast else None,
    )
