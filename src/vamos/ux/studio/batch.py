"""Panel-free hypervolume summaries for the Studio batch comparison.

Hypervolume is computed per problem on objectives normalized with the ideal
and nadir points of the union of all feasible final fronts of that problem in
the batch, with the reference point at 1.1 in every objective. Values are
therefore comparable across algorithms and seeds of one batch, not across
batches or problems.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.quality_indicators.hypervolume import hypervolume

REFERENCE_POINT = 1.1
HV_DESCRIPTION = (
    "Hypervolume of the feasible final front, on objectives normalized per problem with the ideal and nadir "
    f"points of all fronts in this batch; reference point {REFERENCE_POINT} in every objective. Higher is better."
)


@dataclass(frozen=True)
class BatchRun:
    """The final front of one run of a batch."""

    problem: str
    algorithm: str
    seed: int
    F: NDArray[np.float64]
    G: NDArray[np.float64] | None = None

    @property
    def feasible_F(self) -> NDArray[np.float64]:
        if self.G is None or self.G.size == 0:
            return self.F
        mask = np.all(np.asarray(self.G, dtype=float) <= 0.0, axis=1)
        return np.asarray(self.F[mask], dtype=np.float64)


@dataclass(frozen=True)
class BatchFailure:
    """A run, or a whole problem, that could not be completed."""

    problem: str
    algorithm: str
    seed: int | None
    message: str


def run_from_result(problem: str, algorithm: str, seed: int, result: Mapping[str, object]) -> BatchRun:
    F = result.get("F")
    if F is None:
        raise RuntimeError("The run returned no objectives.")
    G = result.get("G")
    return BatchRun(
        problem=problem,
        algorithm=algorithm,
        seed=seed,
        F=np.atleast_2d(np.asarray(F, dtype=float)),
        G=None if G is None else np.atleast_2d(np.asarray(G, dtype=float)),
    )


def normalized_hypervolumes(runs: Sequence[BatchRun]) -> list[float]:
    """Return the normalized hypervolume of every run, in the order given."""
    bounds: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for problem in dict.fromkeys(run.problem for run in runs):
        fronts = [run.feasible_F for run in runs if run.problem == problem and run.feasible_F.size]
        if fronts:
            union = np.vstack(fronts)
            bounds[problem] = (union.min(axis=0), union.max(axis=0))
    values: list[float] = []
    for run in runs:
        points = run.feasible_F
        if not points.size or run.problem not in bounds:
            values.append(0.0)
            continue
        ideal, nadir = bounds[run.problem]
        span = np.where(nadir - ideal > 0.0, nadir - ideal, 1.0)
        normalized = (points - ideal) / span
        values.append(float(hypervolume(normalized, np.full(points.shape[1], REFERENCE_POINT))))
    return values


def summary_rows(runs: Sequence[BatchRun], hv: Sequence[float], failures: Sequence[BatchFailure] = ()) -> list[dict[str, object]]:
    """One row per problem and algorithm, with the best mean hypervolume marked."""
    groups: dict[tuple[str, str], list[float]] = {}
    for run, value in zip(runs, hv, strict=True):
        groups.setdefault((run.problem, run.algorithm), []).append(value)
    failed: dict[tuple[str, str], int] = {}
    for failure in failures:
        if failure.seed is not None:
            failed[(failure.problem, failure.algorithm)] = failed.get((failure.problem, failure.algorithm), 0) + 1
    best: dict[str, float] = {}
    for (problem, _), values in groups.items():
        best[problem] = max(best.get(problem, -np.inf), float(np.mean(values)))
    rows: list[dict[str, object]] = []
    for (problem, algorithm), values in groups.items():
        data = np.asarray(values, dtype=float)
        mean = float(data.mean())
        rows.append(
            {
                "Problem": problem,
                "Algorithm": algorithm,
                "Runs": int(data.size),
                "Failed": failed.get((problem, algorithm), 0),
                "HV mean": mean,
                "HV std": float(data.std(ddof=1)) if data.size > 1 else 0.0,
                "HV median": float(np.median(data)),
                "Best": "yes" if np.isclose(mean, best[problem], rtol=0.0, atol=1e-12) else "",
            }
        )
    return rows


def run_rows(runs: Sequence[BatchRun], hv: Sequence[float]) -> list[dict[str, object]]:
    return [
        {
            "Problem": run.problem,
            "Algorithm": run.algorithm,
            "Seed": run.seed,
            "HV": value,
            "Points": int(run.F.shape[0]),
            "Feasible points": int(run.feasible_F.shape[0]),
        }
        for run, value in zip(runs, hv, strict=True)
    ]


def csv_text(rows: Sequence[Mapping[str, object]]) -> str:
    if not rows:
        return ""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


_LATEX_ESCAPES = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _number(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else float("nan")


def latex_escape(text: str) -> str:
    return "".join(_LATEX_ESCAPES.get(char, char) for char in text)


def summary_latex(rows: Sequence[Mapping[str, object]], *, caption: str = HV_DESCRIPTION, label: str = "tab:vamos-batch") -> str:
    """A booktabs table of the summary; the best mean of each problem is bold."""
    columns = ("Problem", "Algorithm", "Runs", "HV mean", "HV std")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        mean = f"{_number(row['HV mean']):.4f}"
        if row.get("Best"):
            mean = f"\\textbf{{{mean}}}"
        cells = [
            latex_escape(str(row["Problem"])),
            latex_escape(str(row["Algorithm"])),
            str(row["Runs"]),
            mean,
            f"{_number(row['HV std']):.4f}",
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


__all__ = [
    "BatchFailure",
    "BatchRun",
    "HV_DESCRIPTION",
    "REFERENCE_POINT",
    "csv_text",
    "latex_escape",
    "normalized_hypervolumes",
    "run_from_result",
    "run_rows",
    "summary_latex",
    "summary_rows",
]
