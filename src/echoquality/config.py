from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in os.environ:
                return os.environ[name]
            if default is not None:
                return default
            return match.group(0)

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    if not isinstance(content, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return content


def load_config(path: str | Path) -> dict[str, Any]:
    """Load an experiment YAML, optionally merging a referenced base file."""

    config_path = Path(path).resolve()
    config = _read_yaml(config_path)
    extends = config.pop("extends", None)
    if extends:
        base_path = Path(str(extends))
        if not base_path.is_absolute():
            base_path = (config_path.parent / base_path).resolve()
        config = deep_merge(load_config(base_path), config)
    config = _expand_env(config)
    config.setdefault("_config_path", str(config_path))
    return config


def save_resolved_config(config: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean = {key: value for key, value in dict(config).items() if not key.startswith("_")}
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(clean, handle, sort_keys=False, allow_unicode=True)
    return output


def require(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(f"Missing required configuration key: {dotted_key}")
        current = current[part]
    return current
