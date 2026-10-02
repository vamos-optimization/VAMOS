"""Table rows and CSV export for the GUI (UI-independent helpers)."""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from vamos.ux.visualization.interactive import objective_labels

from .catalog import RunSummary

DEFAULT_MAX_VARIABLES = 8
_SIGNIFICANT_DIGITS = 6

Column = dict[str, Any]
Row = dict[str, Any]


def _column(name: str, label: str, *, align: str = "right") -> Column:
    return {"name": name, "label": label, "field": name, "sortable": True, "align": align}


def _format(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(f"{float(value):.{_SIGNIFICANT_DIGITS}g}")
    return str(value)


def run_table(summaries: Sequence[RunSummary]) -> tuple[list[Column], list[Row]]:
    """Return explorer table columns and rows; row ``id`` indexes ``summaries``."""
    columns = [
        _column("problem", "Problem", align="left"),
        _column("algorithm", "Algorithm", align="left"),
        _column("n_objectives", "Objectives"),
        _column("n_solutions", "Solutions"),
        _column("evaluations", "Evaluations"),
        _column("seed", "Seed"),
        _column("status", "Status", align="left"),
        _column("started_at", "Started (UTC)", align="left"),
        _column("relative_path", "Path", align="left"),
    ]
    rows = [
        {
            "id": index,
            "problem": item.problem,
            "algorithm": item.algorithm,
            "n_objectives": item.n_objectives,
            "n_solutions": item.n_solutions,
            "evaluations": item.evaluations,
            "seed": item.seed,
            "status": item.status,
            "started_at": item.started_at.replace("T", " ")[:19],
            "relative_path": item.relative_path,
        }
        for index, item in enumerate(summaries)
    ]
    return columns, rows


def _variables(X: npt.NDArray[Any] | None, n_rows: int) -> npt.NDArray[Any] | None:
    if X is None:
        return None
    arr = np.asarray(X)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr if arr.ndim == 2 and arr.shape[0] == n_rows else None


def solution_table(
    F: npt.NDArray[np.float64],
    X: npt.NDArray[Any] | None,
    indices: Sequence[int],
    *,
    objective_names: Sequence[str] | None = None,
    max_variables: int = DEFAULT_MAX_VARIABLES,
) -> tuple[list[Column], list[Row]]:
    """Return columns and rows for the solutions at ``indices`` (objectives first)."""
    labels = objective_labels(F.shape[1], objective_names)
    variables = _variables(X, F.shape[0])
    n_vars = 0 if variables is None else min(variables.shape[1], max_variables)
    columns = [_column("solution", "#")] + [_column(f"f{m}", label) for m, label in enumerate(labels)]
    columns += [_column(f"x{j}", f"x{j + 1}") for j in range(n_vars)]
    rows: list[Row] = []
    for index in indices:
        if not 0 <= index < F.shape[0]:
            continue
        row: Row = {"id": int(index), "solution": int(index)}
        row.update({f"f{m}": _format(F[index, m]) for m in range(F.shape[1])})
        if variables is not None:
            row.update({f"x{j}": _format(variables[index, j]) for j in range(n_vars)})
        rows.append(row)
    return columns, rows


def solutions_csv(
    F: npt.NDArray[np.float64],
    X: npt.NDArray[Any] | None,
    indices: Sequence[int] | None = None,
    *,
    objective_names: Sequence[str] | None = None,
) -> str:
    """Return CSV text with all objectives and decision variables of the chosen rows."""
    labels = objective_labels(F.shape[1], objective_names)
    variables = _variables(X, F.shape[0])
    chosen = range(F.shape[0]) if indices is None else [i for i in indices if 0 <= i < F.shape[0]]
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    header = ["solution", *labels]
    if variables is not None:
        header += [f"x{j + 1}" for j in range(variables.shape[1])]
    writer.writerow(header)
    for index in chosen:
        row: list[Any] = [int(index), *(repr(float(value)) for value in F[index])]
        if variables is not None:
            row += [_format(value) for value in variables[index]]
        writer.writerow(row)
    return buffer.getvalue()


__all__ = ["DEFAULT_MAX_VARIABLES", "run_table", "solution_table", "solutions_csv"]
