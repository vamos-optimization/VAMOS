"""Sequence-compatible multi-run optimization results."""

from __future__ import annotations

import numbers
from collections.abc import Iterator, Sequence
from typing import Any, overload

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.exceptions import NoSolutionsError, ResultSelectionError

from .model import OptimizationResult


def _lookup_metric_value(run: OptimizationResult, name: str) -> float:
    if not name:
        raise ResultSelectionError("metric name must be a non-empty string.")

    if name in run.data:
        value = run.data[name]
    elif name in run.meta:
        value = run.meta[name]
    else:
        value = _lookup_dotted_metric(run.data, name)
        if value is None:
            value = _lookup_dotted_metric(run.meta, name)
        if value is None:
            raise ResultSelectionError(f"Unknown metric '{name}' for StudyResult aggregation.")

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ResultSelectionError(f"Metric '{name}' is not a numeric scalar and cannot be aggregated.")
    return float(value)


def _lookup_dotted_metric(payload: dict[str, Any], name: str) -> Any | None:
    current: Any = payload
    for part in name.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


class StudyResult(Sequence[OptimizationResult]):
    """Container returned by ``optimize(..., seed=[...])``.

    The object behaves like a read-only sequence of :class:`OptimizationResult`
    while also exposing light-weight aggregation helpers for numeric metrics.
    """

    def __init__(self, runs: Sequence[OptimizationResult] | Iterator[OptimizationResult]):
        self._runs = tuple(runs)

    @property
    def runs(self) -> tuple[OptimizationResult, ...]:
        """Return the underlying immutable run collection."""
        return self._runs

    def __len__(self) -> int:
        return len(self._runs)

    def __iter__(self) -> Iterator[OptimizationResult]:
        return iter(self._runs)

    @overload
    def __getitem__(self, index: int) -> OptimizationResult: ...

    @overload
    def __getitem__(self, index: slice) -> StudyResult: ...

    def __getitem__(self, index: int | slice) -> OptimizationResult | StudyResult:
        if isinstance(index, slice):
            return StudyResult(self._runs[index])
        return self._runs[index]

    def __repr__(self) -> str:
        return f"StudyResult({len(self._runs)} runs)"

    def metric_values(self, name: str) -> NDArray[np.float64]:
        """Return a float array with the named metric extracted from each run."""
        if not self._runs:
            return np.empty(0, dtype=float)
        return np.asarray([_lookup_metric_value(run, name) for run in self._runs], dtype=float)

    def mean(self, name: str) -> float:
        """Return the mean of a numeric metric across runs."""
        values = self.metric_values(name)
        if values.size == 0:
            raise NoSolutionsError("StudyResult contains no runs.")
        return float(np.mean(values))

    def std(self, name: str, ddof: int = 0) -> float:
        """Return the standard deviation of a numeric metric across runs.

        ``ddof=0`` (the default) is the population standard deviation; ``ddof=1`` is the
        sample standard deviation usually reported across seeds.
        """
        values = self.metric_values(name)
        if values.size == 0:
            raise NoSolutionsError("StudyResult contains no runs.")
        return float(np.std(values, ddof=ddof))

    def best_run(self, name: str, maximize: bool = True) -> OptimizationResult:
        """Return the run with the best value for the named metric."""
        values = self.metric_values(name)
        if values.size == 0:
            raise NoSolutionsError("StudyResult contains no runs.")
        best_index = int(np.argmax(values) if maximize else np.argmin(values))
        return self._runs[best_index]


__all__ = ["StudyResult"]
