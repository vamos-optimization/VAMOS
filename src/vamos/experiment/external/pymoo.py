from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np

from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.problem.registry import ProblemSelection
from vamos.foundation.problem.types import ProblemProtocol


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


MetricsBuilder = Callable[[str, str, float, int, np.ndarray[Any, Any]], dict[str, Any]]
Printer = Callable[..., None]


class ExperimentConfig(Protocol):
    population_size: int
    max_evaluations: int
    seed: int


def _run_pymoo_nsga2(
    selection: ProblemSelection,
    *,
    use_native_problem: bool,
    config: ExperimentConfig,
    make_metrics: MetricsBuilder,
    print_banner: Printer,
    print_results: Printer,
) -> dict[str, Any]:
    if selection.spec.key != "zdt1":
        raise ValueError("PyMOO baseline currently supports only ZDT1.")
    problem = selection.instantiate()
    print_banner(problem, selection, "PyMOO NSGA-II", "pymoo")
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem as PymooProblem
        from pymoo.operators.crossover.sbx import SBX
        from pymoo.operators.mutation.pm import PM
        from pymoo.operators.sampling.rnd import FloatRandomSampling
        from pymoo.optimize import minimize
        from pymoo.problems import get_problem
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pymoo is not installed. Install it with 'pip install pymoo' to use this baseline.") from exc

    if use_native_problem:
        pymoo_problem = get_problem("zdt1", n_var=selection.n_var)
    else:

        class _VamosPymooProblem(PymooProblem):  # type: ignore[misc]
            def __init__(self, base_problem: ProblemProtocol) -> None:
                super().__init__(
                    n_var=base_problem.n_var,
                    n_obj=base_problem.n_obj,
                    xl=base_problem.xl,
                    xu=base_problem.xu,
                )
                self._base_problem = base_problem

            def _evaluate(self, X: np.ndarray[Any, Any], out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
                F = np.empty((X.shape[0], self.n_obj))
                self._base_problem.evaluate(X, {"F": F})
                out["F"] = F

        pymoo_problem = _VamosPymooProblem(problem)

    crossover = SBX(prob=0.9, eta=20)
    # pymoo's ``prob`` is the chance of mutating an individual and ``prob_var`` the
    # per-variable rate; 1/n per variable matches the jMetalPy and pygmo baselines.
    mutation = PM(prob=1.0, prob_var=1.0 / selection.n_var, eta=20)
    algorithm = NSGA2(
        pop_size=config.population_size,
        sampling=FloatRandomSampling(),
        crossover=crossover,
        mutation=mutation,
        eliminate_duplicates=True,
    )
    start = time.perf_counter()
    res = minimize(
        pymoo_problem,
        algorithm,
        ("n_eval", config.max_evaluations),
        seed=config.seed,
        verbose=False,
    )
    end = time.perf_counter()
    total_time_ms = (end - start) * 1000.0
    F = np.asarray(res.F, dtype=float)
    metrics = make_metrics("pymoo_nsga2", "pymoo", total_time_ms, config.max_evaluations, F)
    print_results(metrics)
    _logger().info("%s", "=" * 80)
    return metrics


def _run_pymoo_perm_nsga2(
    selection: ProblemSelection,
    *,
    use_native_problem: bool,
    config: ExperimentConfig,
    make_metrics: MetricsBuilder,
    print_banner: Printer,
    print_results: Printer,
) -> dict[str, Any]:
    problem = selection.instantiate()
    encoding = normalize_encoding(getattr(problem, "encoding", "real"))
    if encoding != "permutation":
        raise ValueError("PyMOO permutation baseline requires a permutation-encoded problem.")
    print_banner(problem, selection, "PyMOO NSGA-II (perm)", "pymoo")
    try:
        from pymoo.algorithms.moo.nsga2 import NSGA2
        from pymoo.core.problem import Problem as PymooProblem
        from pymoo.operators.mutation.inversion import InversionMutation
        from pymoo.operators.sampling.rnd import PermutationRandomSampling
        from pymoo.optimize import minimize
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pymoo is not installed. Install it with 'pip install pymoo' to use this baseline.") from exc

    class _VamosPymooPermutationProblem(PymooProblem):  # type: ignore[misc]
        def __init__(self, base_problem: ProblemProtocol) -> None:
            super().__init__(
                n_var=base_problem.n_var,
                n_obj=base_problem.n_obj,
                xl=0,
                xu=base_problem.n_var - 1,
                elementwise_evaluation=False,
            )
            self.base_problem = base_problem

        def _evaluate(self, X: np.ndarray[Any, Any], out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
            perms = np.asarray(X, dtype=int)
            F = np.empty((perms.shape[0], self.n_obj))
            self.base_problem.evaluate(perms, {"F": F})
            out["F"] = F

    pymoo_problem = _VamosPymooPermutationProblem(problem)
    mutation_prob = min(1.0, 2.0 / max(1, problem.n_var))

    def _make_pymoo_perm_crossover(probability: float) -> Any:
        try:
            from pymoo.operators.crossover.pmx import PMX

            return PMX(prob=probability)
        except ImportError:
            try:
                from pymoo.operators.crossover.ox import OrderCrossover
            except ImportError as exc:
                raise ImportError(
                    "pymoo permutation crossover operators are unavailable; upgrade pymoo to a version that ships PMX or OX."
                ) from exc

            class _OrderCrossoverWrapper(OrderCrossover):  # type: ignore[misc]
                def __init__(self, prob: float) -> None:
                    super().__init__(prob=prob)

            return _OrderCrossoverWrapper(probability)

    crossover = _make_pymoo_perm_crossover(probability=0.9)
    algorithm = NSGA2(
        pop_size=config.population_size,
        sampling=PermutationRandomSampling(),
        crossover=crossover,
        mutation=InversionMutation(prob=mutation_prob),
        eliminate_duplicates=True,
    )
    start = time.perf_counter()
    res = minimize(
        pymoo_problem,
        algorithm,
        ("n_eval", config.max_evaluations),
        seed=config.seed,
        verbose=False,
    )
    end = time.perf_counter()
    total_time_ms = (end - start) * 1000.0
    F = np.asarray(res.F, dtype=float)
    metrics = make_metrics("pymoo_perm_nsga2", "pymoo", total_time_ms, config.max_evaluations, F)
    print_results(metrics)
    _logger().info("%s", "=" * 80)
    return metrics
