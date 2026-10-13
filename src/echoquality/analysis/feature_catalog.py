from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    level: str
    family: str
    formula: str
    interpretation: str
    support: str
    units: str = "dimensionless"
    source_type: str = "custom"
    source: str = "EchoQuality 2.0 implementation"
    parameters: dict[str, object] = field(default_factory=dict)
    known_confounders: tuple[str, ...] = ()
    higher_is_better: bool | None = None
    aliases: tuple[str, ...] = ()


class FeatureCatalog:
    def __init__(self, definitions: Iterable[FeatureDefinition] = ()) -> None:
        self._definitions: dict[str, FeatureDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: FeatureDefinition) -> None:
        if definition.name in self._definitions and self._definitions[definition.name] != definition:
            raise ValueError(f"Feature already registered with a different definition: {definition.name}")
        self._definitions[definition.name] = definition

    def extend(self, definitions: Iterable[FeatureDefinition]) -> None:
        for definition in definitions:
            self.register(definition)

    def get(self, name: str) -> FeatureDefinition:
        return self._definitions[name]

    def level(self, name: str) -> str:
        return self.get(name).level

    def definitions(self) -> list[FeatureDefinition]:
        return list(self._definitions.values())

    def names(self, level: str | None = None) -> list[str]:
        return [name for name, definition in self._definitions.items() if level is None or definition.level == level]

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for definition in self._definitions.values():
            row = asdict(definition)
            row["parameters"] = repr(row["parameters"])
            row["known_confounders"] = "; ".join(row["known_confounders"])
            row["aliases"] = "; ".join(row["aliases"])
            rows.append(row)
        return pd.DataFrame(rows).sort_values(["level", "family", "name"]).reset_index(drop=True)

    def save_csv(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.to_frame().to_csv(output, index=False)
        return output

    def validate_values(self, values: dict[str, float]) -> None:
        unknown = set(values) - set(self._definitions)
        if unknown:
            raise ValueError(f"Unregistered feature values: {sorted(unknown)}")


def definition(
    name: str,
    level: str,
    family: str,
    formula: str,
    interpretation: str,
    support: str,
    *,
    units: str = "dimensionless",
    source_type: str = "custom",
    source: str = "EchoQuality 2.0 implementation",
    parameters: dict[str, object] | None = None,
    confounders: tuple[str, ...] = (),
    higher_is_better: bool | None = None,
    aliases: tuple[str, ...] = (),
) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        level=level,
        family=family,
        formula=formula,
        interpretation=interpretation,
        support=support,
        units=units,
        source_type=source_type,
        source=source,
        parameters=parameters or {},
        known_confounders=confounders,
        higher_is_better=higher_is_better,
        aliases=aliases,
    )
