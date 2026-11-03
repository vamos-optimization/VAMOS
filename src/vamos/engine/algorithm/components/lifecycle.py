"""
Lifecycle helpers for algorithm runs (initial population, evaluation strategies).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.population import (
    DecisionArray,
    FloatArray,
    initialize_evaluated_population,
    resolve_bounds,
)
from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.eval.backends import EvaluationBackend, SerialEvalBackend

if TYPE_CHECKING:
    from vamos.foundation.problem.types import ProblemProtocol


def evaluate_batch(
    problem: ProblemProtocol,
    eval_strategy: EvaluationBackend,
    X: DecisionArray,
    constraint_mode: str,
) -> tuple[FloatArray, FloatArray | None]:
    """
    Evaluate a decision batch through the configured backend.

    Parameters
    ----------
    problem : ProblemProtocol
        The optimization problem.
    eval_strategy : EvaluationBackend
        Backend for evaluating solutions.
    X : np.ndarray
        Decision vectors to evaluate.
    constraint_mode : str
        Constraint handling mode.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        Objective and optional constraint arrays.
    """
    eval_result = eval_strategy.evaluate(X, problem)
    F = eval_result.F
    G = eval_result.G if constraint_mode != "none" else None
    return F, G


def setup_initial_population(
    problem: ProblemProtocol,
    eval_strategy: EvaluationBackend,
    rng: np.random.Generator,
    pop_size: int,
    constraint_mode: str,
    initializer_cfg: dict[str, Any] | None = None,
    *,
    max_evaluations: int | None = None,
    algorithm: str = "The algorithm",
) -> tuple[DecisionArray, FloatArray, FloatArray | None, int]:
    """
    Initialize and evaluate the starting population.

    Parameters
    ----------
    problem : ProblemProtocol
        The optimization problem.
    eval_strategy : EvaluationBackend
        Backend for evaluating solutions.
    rng : np.random.Generator
        Random number generator.
    pop_size : int
        Population size.
    constraint_mode : str
        Constraint handling mode ('none', 'feasibility', etc.).
    initializer_cfg : dict[str, Any] | None
        Optional initializer configuration.
    max_evaluations : int | None
        Evaluation budget of the run; the opposition-based initializer needs ``2 * pop_size``.
    algorithm : str
        Algorithm name for error messages.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray | None, int]
        (X, F, G, n_eval) - decision variables, objectives, constraints, and every
        evaluation the initialization took.
    """
    encoding = normalize_encoding(getattr(problem, "encoding", "real"))
    xl, xu = resolve_bounds(problem, encoding)
    return initialize_evaluated_population(
        pop_size,
        problem.n_var,
        xl,
        xu,
        encoding,
        rng,
        problem,
        initializer_cfg,
        evaluate=lambda X: evaluate_batch(problem, eval_strategy, X, constraint_mode),
        max_evaluations=max_evaluations,
        algorithm=algorithm,
    )


def get_eval_strategy(
    eval_strategy: EvaluationBackend | None,
) -> EvaluationBackend:
    """
    Get evaluation backend, defaulting to serial.

    Parameters
    ----------
    eval_strategy : EvaluationBackend | None
        User-provided backend or None.

    Returns
    -------
    EvaluationBackend
        The backend or a serial implementation.
    """
    return eval_strategy or SerialEvalBackend()


__all__ = ["setup_initial_population", "get_eval_strategy", "evaluate_batch"]
