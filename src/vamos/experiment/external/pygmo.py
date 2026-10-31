from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.problem.registry import ProblemSelection


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _pygmo_seed(seed: int) -> int:
    """Return ``seed`` as the unsigned 32-bit integer pygmo's constructors accept.

    Seeds in ``[0, 2**32)`` pass unchanged, so existing runs keep their streams. Wider
    seeds (the CLI draws 64-bit seeds when ``--seed`` is omitted) are folded by XOR of
    the 32-bit words of their magnitude, so every bit still affects the result.
    """
    if 0 <= seed <= 0xFFFF_FFFF:
        return seed
    magnitude = abs(seed)
    folded = 0
    while magnitude:
        folded ^= magnitude & 0xFFFF_FFFF
        magnitude >>= 32
    return folded


def _run_pygmo_nsga2(
    selection: ProblemSelection,
    *,
    use_native_problem: bool,
    config: ExperimentConfig,
    make_metrics: Callable[[str, str, float, int, np.ndarray[Any, Any]], dict[str, Any]],
    print_banner: Callable[[Any, ProblemSelection, str, str], None],
    print_results: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    if selection.spec.key != "zdt1":
        raise ValueError("PyGMO baseline currently supports only ZDT1.")
    problem = selection.instantiate()
    print_banner(problem, selection, "PyGMO NSGA-II", "pygmo")
    try:
        import pygmo as pg
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pygmo is not installed. Install it with 'pip install pygmo' to use this baseline.") from exc

    generations = config.max_evaluations // config.population_size - 1
    seed = _pygmo_seed(int(config.seed))
    uda = pg.nsga2(
        gen=generations,
        seed=seed,
        cr=0.9,
        eta_c=20.0,
        m=1.0 / selection.n_var,
        eta_m=20.0,
    )
    algo = pg.algorithm(uda)
    if use_native_problem:
        # pygmo names the ZDT dimension ``param``: zdt(prob_id=1, param=30).
        base_problem = pg.zdt(prob_id=1, param=selection.n_var)
    else:

        class _VamosPyGMOProblem:
            def __init__(self, base_problem: Any) -> None:
                self._base_problem = base_problem
                self._lower = [base_problem.xl] * base_problem.n_var
                self._upper = [base_problem.xu] * base_problem.n_var

            def fitness(self, x: Any) -> list[float]:
                X = np.asarray(x, dtype=float)[np.newaxis, :]
                F = np.empty((1, self._base_problem.n_obj))
                self._base_problem.evaluate(X, {"F": F})
                return [float(v) for v in F[0]]

            def get_bounds(self) -> tuple[list[float], list[float]]:
                return (self._lower, self._upper)

            def get_nobj(self) -> int:
                return int(self._base_problem.n_obj)

            def get_name(self) -> str:
                return "VAMOS-ZDT1"

        base_problem = _VamosPyGMOProblem(problem)

    pg_problem = pg.problem(base_problem)
    pop = pg.population(pg_problem, size=config.population_size, seed=seed)
    start = time.perf_counter()
    pop = algo.evolve(pop)
    end = time.perf_counter()
    total_eval = config.population_size * (generations + 1)
    total_time_ms = (end - start) * 1000.0
    F = np.asarray(pop.get_f(), dtype=float)
    metrics = make_metrics("pygmo_nsga2", "pygmo", total_time_ms, total_eval, F)
    print_results(metrics)
    _logger().info("%s", "=" * 80)
    return metrics
