"""Panel-free statistical comparison of the algorithms of one study for VAMOS Studio.

Indicators are computed per run from the study's canonical runs: the
normalized hypervolume of the batch (higher is better) and, where VAMOS ships a
reference front with the problem's number of objectives, IGD+ on objectives
normalized by that front (lower is better). A run without a feasible solution
takes the worst value of each indicator, HV = 0 and IGD+ = +inf, and stays in
the tests, effect sizes, medians and ranks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.problem.resolver import resolve_reference_front_path
from vamos.ux.studio.algorithm_catalog import algorithm_label
from vamos.ux.studio.batch import BatchRun, normalized_hypervolumes
from vamos.ux.studio.comparison_stats import (
    a12,
    a12_magnitude,
    adjust,
    average_ranks,
    cd_groups,
    friedman_p,
    holm_against_pivot,
    nemenyi_cd,
    rank_sum_p,
    verdict,
)
from vamos.ux.studio.data import RunRecord, load_runs_from_study

HV = "HV"
IGD_PLUS = "IGD+"
INDICATOR_DIRECTIONS: dict[str, bool] = {HV: True, IGD_PLUS: False}


@dataclass(frozen=True)
class ComparisonData:
    """Per-run indicator values and costs of one study, by problem and algorithm."""

    root: Path
    problems: tuple[str, ...]
    algorithms: tuple[str, ...]
    runs: tuple[BatchRun, ...]
    values: Mapping[str, Mapping[tuple[str, str], NDArray[np.float64]]]
    runtimes_ms: Mapping[tuple[str, str], NDArray[np.float64]] = field(default_factory=dict)

    @property
    def indicators(self) -> tuple[str, ...]:
        return tuple(name for name in (HV, IGD_PLUS) if any(len(v) for v in self.values.get(name, {}).values()))

    def sample(self, indicator: str, problem: str, algorithm: str) -> NDArray[np.float64]:
        """The indicator value of every run, +inf included; only undefined (NaN) values are left out."""
        values = self.values.get(indicator, {}).get((problem, algorithm), np.empty(0))
        return np.asarray(values[~np.isnan(values)], dtype=np.float64)

    def feasible_runs(self, problem: str, algorithm: str) -> tuple[int, int]:
        """The runs of ``algorithm`` on ``problem`` with at least one feasible solution, and all its runs."""
        runs = [run for run in self.runs if run.problem == problem and run.algorithm == algorithm]
        return sum(1 for run in runs if run.feasible_F.size), len(runs)


@dataclass(frozen=True)
class Comparison:
    """One contender against the pivot on one problem."""

    problem: str
    contender: str
    pivot: str
    median: float
    pivot_median: float
    p_value: float
    p_adjusted: float
    effect: float
    magnitude: str
    verdict: str


@dataclass(frozen=True)
class Ranking:
    """Average ranks over problems and the tests built on them."""

    algorithms: tuple[str, ...]
    ranks: tuple[float, ...]
    problems: int
    friedman: tuple[float, float] | None
    cd: float | None
    groups: tuple[tuple[int, int], ...]
    against_pivot: tuple[tuple[float, float, float], ...]
    warnings: tuple[str, ...]


def igd_plus(points: NDArray[np.float64], reference: NDArray[np.float64]) -> float:
    """Mean distance from every reference point to the closest solution, counting only shortfalls.

    A run without feasible solutions gets +inf, the worst value, so that it still counts in the tests.
    """
    if points.size == 0:
        return float("inf")
    shortfall = np.maximum(points[None, :, :] - reference[:, None, :], 0.0)
    return float(np.sqrt((shortfall**2).sum(axis=2)).min(axis=1).mean())


def load_comparison_data(root: Path, *, cache: dict[str, RunRecord] | None = None) -> ComparisonData:
    """Read the study's successful runs and compute their indicators."""
    records = [run for run in load_runs_from_study(root, cache=cache) if np.asarray(run.fun).size]
    runtimes: dict[tuple[str, str], list[float]] = {}
    for record in records:
        runtime = record.metadata.get("study_summary", {}).get("runtime_ms")
        if isinstance(runtime, (int, float)):
            runtimes.setdefault((record.problem_name, algorithm_label(record.algorithm_name)), []).append(float(runtime))
    runs = tuple(
        BatchRun(
            record.problem_name,
            algorithm_label(record.algorithm_name),
            record.seed,
            np.atleast_2d(np.asarray(record.fun, dtype=float)),
            record.constraints,
        )
        for record in records
    )
    problems = tuple(dict.fromkeys(run.problem for run in runs))
    algorithms = tuple(dict.fromkeys(run.algorithm for run in runs))
    hv = normalized_hypervolumes(runs)
    values: dict[str, dict[tuple[str, str], list[float]]] = {HV: {}, IGD_PLUS: {}}
    references = {problem: _reference_front(problem, runs) for problem in problems}
    for run, hv_value in zip(runs, hv, strict=True):
        values[HV].setdefault((run.problem, run.algorithm), []).append(hv_value)
        reference = references[run.problem]
        if reference is not None:
            ideal, span = reference[1], reference[2]
            normalized = (run.feasible_F - ideal) / span
            values[IGD_PLUS].setdefault((run.problem, run.algorithm), []).append(igd_plus(normalized, (reference[0] - ideal) / span))
    return ComparisonData(
        root=Path(root),
        problems=problems,
        algorithms=algorithms,
        runs=runs,
        values={name: {key: np.asarray(v, dtype=float) for key, v in by_key.items()} for name, by_key in values.items()},
        runtimes_ms={key: np.asarray(v, dtype=float) for key, v in runtimes.items()},
    )


def compare_with_pivot(data: ComparisonData, indicator: str, pivot: str, *, correction: str, alpha: float) -> list[Comparison]:
    """Every contender against the pivot on every problem, with corrected p-values."""
    higher = INDICATOR_DIRECTIONS[indicator]
    cells: list[tuple[str, str, NDArray[np.float64], NDArray[np.float64]]] = []
    for problem in data.problems:
        reference = data.sample(indicator, problem, pivot)
        for contender in data.algorithms:
            sample = data.sample(indicator, problem, contender)
            if contender != pivot and sample.size and reference.size:
                cells.append((problem, contender, sample, reference))
    p_values = [rank_sum_p(sample, reference) for _, _, sample, reference in cells]
    adjusted = adjust(p_values, correction)
    comparisons: list[Comparison] = []
    for (problem, contender, sample, reference), p, p_adj in zip(cells, p_values, adjusted, strict=True):
        effect = a12(sample, reference, higher_is_better=higher)
        comparisons.append(
            Comparison(
                problem=problem,
                contender=contender,
                pivot=pivot,
                median=float(np.median(sample)),
                pivot_median=float(np.median(reference)),
                p_value=p,
                p_adjusted=float(p_adj),
                effect=effect,
                magnitude=a12_magnitude(effect),
                verdict=verdict(float(p_adj), effect, alpha),
            )
        )
    return comparisons


def infeasible_note(data: ComparisonData, indicator: str) -> str:
    """Say how many runs found no feasible solution and how they count; empty when every run found one."""
    keys = data.values.get(indicator, {})
    runs = [run for run in data.runs if (run.problem, run.algorithm) in keys]
    infeasible = sum(1 for run in runs if not run.feasible_F.size)
    if not infeasible:
        return ""
    worst = "0" if INDICATOR_DIRECTIONS[indicator] else "∞"
    return (
        f"{infeasible} of {len(runs)} runs found no feasible solution. They count as the worst value, "
        f"{indicator} = {worst}, in the tests, effect sizes, medians and ranks; the tables give the feasible runs of every cell."
    )


def verdict_counts(comparisons: Sequence[Comparison]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for item in comparisons:
        counts.setdefault(item.contender, {"+": 0, "=": 0, "-": 0})[item.verdict] += 1
    return counts


def median_matrix(data: ComparisonData, indicator: str) -> tuple[tuple[str, ...], NDArray[np.float64]]:
    """Problems with a value for every algorithm, and their medians (problems x algorithms)."""
    rows: list[list[float]] = []
    kept: list[str] = []
    for problem in data.problems:
        samples = [data.sample(indicator, problem, algorithm) for algorithm in data.algorithms]
        if all(sample.size for sample in samples):
            kept.append(problem)
            rows.append([float(np.median(sample)) for sample in samples])
    return tuple(kept), np.asarray(rows, dtype=float).reshape(len(rows), len(data.algorithms))


def rank_algorithms(data: ComparisonData, indicator: str, pivot: str, *, alpha: float = 0.05) -> Ranking:
    problems, matrix = median_matrix(data, indicator)
    k, n = len(data.algorithms), len(problems)
    warnings: list[str] = []
    if n == 0:
        return Ranking(data.algorithms, (), 0, None, None, (), (), ("No problem has results for every algorithm.",))
    ranks = average_ranks(matrix, higher_is_better=INDICATOR_DIRECTIONS[indicator])
    if n < 5:
        warnings.append(f"Only {n} problem(s): rank-based tests over problems have little power; read them as indicative.")
    if k < 3:
        warnings.append("The Friedman test needs at least three algorithms.")
    if note := infeasible_note(data, indicator):
        warnings.append(note)
    cd = nemenyi_cd(k, n, alpha) if k >= 2 else None
    groups = tuple(cd_groups(list(ranks), cd)) if cd is not None else ()
    against = tuple(holm_against_pivot(ranks, data.algorithms.index(pivot), n)) if pivot in data.algorithms else ()
    return Ranking(data.algorithms, tuple(float(r) for r in ranks), n, friedman_p(matrix), cd, groups, against, tuple(warnings))


def median_run(data: ComparisonData, indicator: str, problem: str, algorithm: str) -> BatchRun | None:
    """The run whose indicator value is the median of its problem and algorithm."""
    candidates = [run for run in data.runs if run.problem == problem and run.algorithm == algorithm]
    values = data.values.get(indicator, {}).get((problem, algorithm))
    if not candidates or values is None or len(values) != len(candidates):
        return candidates[0] if candidates else None
    defined = [index for index, value in enumerate(values) if not np.isnan(value)]
    if not defined:
        return candidates[0]
    ordered = sorted(defined, key=lambda index: float(values[index]))
    return candidates[ordered[(len(ordered) - 1) // 2]]


def _reference_front(problem: str, runs: Sequence[BatchRun]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]] | None:
    n_obj = next((run.F.shape[1] for run in runs if run.problem == problem), None)
    path = resolve_reference_front_path(problem, None, n_obj=n_obj)
    if path is None or n_obj is None:
        return None
    try:
        front = np.atleast_2d(np.loadtxt(path, delimiter=","))
    except (OSError, ValueError):
        return None
    if front.shape[1] != n_obj:
        return None
    ideal, nadir = front.min(axis=0), front.max(axis=0)
    span = np.where(nadir - ideal > 0, nadir - ideal, 1.0)
    return front, ideal, span


__all__ = [
    "HV",
    "IGD_PLUS",
    "INDICATOR_DIRECTIONS",
    "Comparison",
    "ComparisonData",
    "Ranking",
    "compare_with_pivot",
    "igd_plus",
    "infeasible_note",
    "load_comparison_data",
    "median_matrix",
    "median_run",
    "rank_algorithms",
    "verdict_counts",
]
