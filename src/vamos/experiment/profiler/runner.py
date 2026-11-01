"""
Performance profiling runner for comparing backends.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from vamos.experiment.runtime.catalog import EXPERIMENT_BACKENDS
from vamos.experiment.unified import optimize
from vamos.foundation.constraints.utils import is_feasible
from vamos.foundation.core.hv_stop import near_nadir_reference
from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.problem.resolver import resolve_reference_front_path
from vamos.foundation.quality_indicators.hypervolume import hypervolume

# Engines ``vamos profile`` accepts: the kernel backends and the automatic choice, as in ``vamos --engine``.
PROFILE_ENGINES = (*EXPERIMENT_BACKENDS, "auto")
_CSV_COLUMNS = ["engine", "time_seconds", "time_min", "time_max", "repeats", "n_solutions", "hypervolume", "error"]


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Timings of one engine: the median and range of its timed runs, which follow one warm-up run.

    A failed engine has ``error`` set and no timings, solution count or hypervolume.
    """

    engine: str
    time_seconds: float | None
    n_solutions: int | None
    hypervolume: float | None = None
    time_min: float | None = None
    time_max: float | None = None
    repeats: int = 0
    error: str | None = None


@dataclass
class ProfileReport:
    """Aggregated profiling report."""

    problem: str
    budget: int
    results: list[ProfileResult] = field(default_factory=list)
    repeats: int = 1

    @property
    def failed(self) -> list[ProfileResult]:
        """The engines whose runs failed."""
        return [result for result in self.results if result.error is not None]

    def print_summary(self) -> None:
        """Print formatted summary table."""
        log = _logger()
        log.info("%s", "=" * 64)
        log.info("VAMOS Performance Profile")
        log.info("%s", "=" * 64)
        log.info("Problem: %s", self.problem)
        log.info("Budget: %s evaluations", self.budget)
        log.info("Time: median of %s timed run(s) per engine, after one warm-up run", self.repeats)
        log.info("%-12s %-10s %-15s %-10s %-10s", "Engine", "Time (s)", "Range (s)", "Solutions", "HV")
        log.info("%s", "-" * 64)
        for r in self.results:
            if r.error is not None:
                log.info("%-12s failed: %s", r.engine, r.error)
                continue
            hv_str = "N/A" if r.hypervolume is None else f"{r.hypervolume:.4f}"
            spread = f"{r.time_min:.3f}-{r.time_max:.3f}"
            log.info("%-12s %-10.3f %-15s %-10s %-10s", r.engine, r.time_seconds, spread, r.n_solutions, hv_str)

        # Speedups against the first engine that ran.
        timed = [(r.engine, seconds) for r in self.results if (seconds := r.time_seconds)]
        if len(timed) > 1:
            base_engine, base_time = timed[0]
            log.info("Speedup vs %s:", base_engine)
            for engine, seconds in timed[1:]:
                log.info("  %s: %.1fx", engine, base_time / seconds)

    def to_csv(self, path: str) -> None:
        """Export results to CSV; a value the table shows as missing is an empty cell."""
        import csv

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(_CSV_COLUMNS)
            for r in self.results:
                row = [r.engine, r.time_seconds, r.time_min, r.time_max, r.repeats or None, r.n_solutions, r.hypervolume, r.error]
                writer.writerow(["" if value is None else value for value in row])


def check_engines(engines: Sequence[str]) -> None:
    """Reject engine names that ``vamos profile`` cannot run, before any run starts.

    Raises
    ------
    ValueError
        If ``engines`` is empty or names an unknown engine.
    """
    if not engines:
        raise ValueError("At least one engine is required.")
    unknown = [engine for engine in engines if engine not in PROFILE_ENGINES]
    if unknown:
        raise ValueError(f"Unknown engine(s): {', '.join(unknown)}. Choose from: {', '.join(PROFILE_ENGINES)}.")


def _hv_reference(problem: str, explicit: Sequence[float] | None) -> np.ndarray[Any, Any] | None:
    """The hypervolume reference point: the explicit one, or the near-nadir point of the problem's reference front."""
    n_obj = int(make_problem_selection(problem).instantiate().n_obj)
    if explicit is not None:
        ref = np.asarray(explicit, dtype=float)
        if ref.shape != (n_obj,) or not np.isfinite(ref).all():
            raise ValueError(
                f"The hypervolume reference needs one finite value per objective: {problem} has {n_obj} objectives, got {ref.size} values."
            )
        return ref
    path = resolve_reference_front_path(problem, None, n_obj=n_obj)
    front = None if path is None else np.loadtxt(path, delimiter=",", ndmin=2)
    if front is None or front.shape[1] != n_obj:
        reason = "no reference front is packaged" if front is None else f"its reference front has {front.shape[1]} objectives, not {n_obj}"
        _logger().warning("Hypervolume not computed for %s: %s. Pass --hv-reference to set the reference point.", problem, reason)
        return None
    return near_nadir_reference(front)


def _hypervolume(result: Any, ref: np.ndarray[Any, Any], engine: str) -> float | None:
    """Hypervolume of the feasible solutions that the reference point dominates; the others do not count."""
    F = np.asarray(result.F, dtype=float)
    data = getattr(result, "data", None)
    G = data.get("G") if isinstance(data, dict) else None
    if G is not None:
        F = F[is_feasible(np.asarray(G, dtype=float), n=F.shape[0])]
    try:
        return hypervolume(F[np.all(F <= ref, axis=1)], ref)
    except ValueError as exc:
        _logger().warning("Hypervolume not computed for engine %s: %s", engine, exc)
        return None


def _profile_engine(problem: str, engine: str, budget: int, seed: int, repeats: int, ref: np.ndarray[Any, Any] | None) -> ProfileResult:
    _logger().info("Profiling %s...", engine)
    times: list[float] = []
    try:
        # The warm-up run pays one-time costs (Numba compilation, lazy imports) and is not timed.
        result = optimize(problem, engine=engine, max_evaluations=budget, seed=seed, verbose=False)
        for _ in range(repeats):
            start = time.perf_counter()
            result = optimize(problem, engine=engine, max_evaluations=budget, seed=seed, verbose=False)
            times.append(time.perf_counter() - start)
    except Exception as exc:
        _logger().warning("Engine %s failed: %s", engine, exc)
        return ProfileResult(engine=engine, time_seconds=None, n_solutions=None, error=f"{type(exc).__name__}: {exc}")
    return ProfileResult(
        engine=engine,
        time_seconds=statistics.median(times),
        n_solutions=len(result),
        hypervolume=None if ref is None else _hypervolume(result, ref, engine),
        time_min=min(times),
        time_max=max(times),
        repeats=repeats,
    )


def run_profile(
    problem: str,
    engines: Sequence[str] = ("numpy",),
    budget: int = 2000,
    seed: int = 42,
    compute_hv: bool = True,
    *,
    repeats: int = 3,
    hv_reference: Sequence[float] | None = None,
) -> ProfileReport:
    """
    Profile optimization across multiple backends.

    Each engine runs once untimed, to pay one-time costs such as Numba compilation, and then
    ``repeats`` timed runs; the report gives their median and range.

    Parameters
    ----------
    problem : str
        Problem name.
    engines : Sequence[str], default ``("numpy",)``
        Engines to compare, from ``PROFILE_ENGINES``.
    budget : int, default ``2000``
        Evaluation budget per run.
    seed : int, default ``42``
        Random seed for reproducibility.
    compute_hv : bool, default ``True``
        Whether to compute hypervolume.
    repeats : int, default ``3``
        Timed runs per engine, after the warm-up run.
    hv_reference : Sequence[float] or None, optional
        Hypervolume reference point. By default, the nadir of the problem's reference front plus
        10 % of its range; without a reference front, hypervolume is not computed.

    Returns
    -------
    ProfileReport
        Timing and quality report for the requested engines. A failed engine is recorded with
        its error and no timings.

    Raises
    ------
    ValueError
        If an engine is unknown, ``repeats`` is below 1, or ``hv_reference`` does not have one
        value per objective.
    """
    check_engines(engines)
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")
    ref = _hv_reference(problem, hv_reference) if compute_hv else None
    report = ProfileReport(problem=problem, budget=budget, repeats=repeats)
    for engine in engines:
        report.results.append(_profile_engine(problem, engine, budget, seed, repeats, ref))
    return report


__all__ = ["PROFILE_ENGINES", "ProfileReport", "ProfileResult", "check_engines", "run_profile"]
