from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from echoquality.experiments.paths import build_experiment_paths
from echoquality.experiments.stages import STAGE_FUNCTIONS, stage_quality_gate

STAGE_ORDER: tuple[str, ...] = (
    "manifest",
    "features",
    "indicators",
    "gate",
    "references",
    "diagnostics",
    "processing",
    "report",
)

DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "manifest": (),
    "features": ("manifest",),
    "indicators": ("features",),
    "gate": ("features",),
    "references": ("features",),
    "diagnostics": ("gate", "references"),
    "processing": ("diagnostics", "references"),
    "report": ("indicators", "gate"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_state(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"created_at": _utc_now(), "stages": {}}


def _write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _utc_now()
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def resolve_stages(requested: Iterable[str] | None) -> list[str]:
    if not requested:
        return list(STAGE_ORDER)
    names = list(requested)
    if "all" in names:
        return list(STAGE_ORDER)
    unknown = set(names) - set(STAGE_ORDER)
    if unknown:
        raise ValueError(f"Unknown stages: {sorted(unknown)}")
    required: set[str] = set()

    def add_with_dependencies(stage: str) -> None:
        for dependency in DEPENDENCIES[stage]:
            add_with_dependencies(dependency)
        required.add(stage)

    for name in names:
        add_with_dependencies(name)
    return [stage for stage in STAGE_ORDER if stage in required]


def run_experiment(
    config: dict[str, Any],
    *,
    stages: Iterable[str] | None = None,
    resume: bool = True,
    force: bool = False,
    fast: bool | None = None,
) -> dict[str, Any]:
    paths = build_experiment_paths(config)
    paths.root.mkdir(parents=True, exist_ok=True)
    state = _read_state(paths.state_file)
    state["experiment"] = config.get("experiment", {}).get("name")
    selected = resolve_stages(stages)
    for stage in selected:
        existing = state.get("stages", {}).get(stage, {})
        if resume and not force and existing.get("status") == "completed":
            print(f"[SKIP] {stage}: already completed")
            continue
        print(f"[RUN ] {stage}")
        state.setdefault("stages", {})[stage] = {"status": "running", "started_at": _utc_now()}
        _write_state(paths.state_file, state)
        try:
            if stage == "gate":
                result = stage_quality_gate(config, fast_override=fast)
            else:
                result = STAGE_FUNCTIONS[stage](config)
            state["stages"][stage] = {
                "status": "completed",
                "started_at": state["stages"][stage]["started_at"],
                "completed_at": _utc_now(),
                "result": result,
            }
            _write_state(paths.state_file, state)
            print(f"[DONE] {stage}")
        except Exception as exc:
            state["stages"][stage] = {
                "status": "failed",
                "started_at": state["stages"][stage]["started_at"],
                "failed_at": _utc_now(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            _write_state(paths.state_file, state)
            print(f"[FAIL] {stage}: {exc}")
            raise
    return state
