from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import pandas as pd


@dataclass(frozen=True)
class KernelSearchResult:
    best_parameter: float | int | None
    table: pd.DataFrame


def optimize_scalar_parameter(
    values: Iterable[float | int],
    evaluator: Callable[[float | int], dict[str, float]],
    *,
    objective_name: str = "objective",
) -> KernelSearchResult:
    """Generic one-dimensional parameter sensitivity experiment.

    The evaluator must return an objective plus safety metrics. This function
    intentionally avoids visual/manual selection of the best kernel.
    """

    rows = []
    for value in values:
        metrics = dict(evaluator(value))
        rows.append({"parameter": value, **metrics})
    table = pd.DataFrame(rows).sort_values(objective_name, ascending=False).reset_index(drop=True)
    best = table.iloc[0]["parameter"] if not table.empty else None
    return KernelSearchResult(best_parameter=best, table=table)
