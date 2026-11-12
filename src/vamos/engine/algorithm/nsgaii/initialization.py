"""
Setup and initialization helpers for NSGA-II.

This module initializes the population, genealogy tracking and parent selection.
Termination parsing and archives use the shared helpers in
``components/termination.py`` and ``engine/archive/factory.py``.

Operator pool building lives in operators/policies/nsgaii.py.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vamos.engine.algorithm.components.lifecycle import setup_initial_population
from vamos.engine.hooks.genealogy import DefaultGenealogyTracker, GenealogyTracker
from vamos.foundation.eval.backends import EvaluationBackend
from vamos.foundation.problem.types import ProblemProtocol

# Constants
DEFAULT_TOURNAMENT_PRESSURE = 2


def setup_population(
    problem: ProblemProtocol,
    eval_strategy: EvaluationBackend,
    rng: np.random.Generator,
    pop_size: int,
    constraint_mode: str,
    initializer_cfg: dict[str, Any] | None = None,
    max_evaluations: int | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any] | None, int]:
    """Initialize and evaluate the starting population.

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

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray | None, int]
        (X, F, G, n_eval) - decision variables, objectives, constraints, and every
        evaluation the initialization took.
    """
    return setup_initial_population(
        problem, eval_strategy, rng, pop_size, constraint_mode, initializer_cfg, max_evaluations=max_evaluations, algorithm="NSGA-II"
    )


def setup_genealogy(
    pop_size: int,
    F: np.ndarray[Any, Any],
    track_genealogy: bool,
) -> tuple[GenealogyTracker | None, np.ndarray[Any, Any] | None]:
    """Initialize genealogy tracking if enabled.

    Parameters
    ----------
    pop_size : int
        Population size.
    F : np.ndarray
        Initial fitness values.
    track_genealogy : bool
        Whether to enable genealogy tracking.

    Returns
    -------
    tuple[GenealogyTracker | None, np.ndarray | None]
        (genealogy_tracker, individual_ids)
    """
    if not track_genealogy:
        return None, None

    genealogy_tracker = DefaultGenealogyTracker()
    ids = np.arange(pop_size, dtype=int)
    for i in range(pop_size):
        genealogy_tracker.new_individual(
            generation=0,
            parents=[],
            operator_name=None,
            algorithm_name="nsgaii",
            fitness=F[i] if F is not None and i < F.shape[0] else None,
        )
    return genealogy_tracker, ids


def setup_selection(
    sel_method: str,
    sel_params: dict[str, Any],
) -> tuple[str, int]:
    """Parse selection config and return (method, tournament_size).

    Parameters
    ----------
    sel_method : str
        Selection method name ('tournament' or 'random').
    sel_params : dict[str, Any]
        Selection parameters.

    Returns
    -------
    tuple[str, int]
        (method, tournament_size)

    Raises
    ------
    ValueError
        If selection method is not supported.
    """
    allowed = ("tournament", "random", "boltzmann", "ranking", "sus")
    if sel_method not in allowed:
        raise ValueError(f"Unsupported selection method '{sel_method}'. Must be one of {allowed}.")
    if sel_method == "tournament":
        unexpected = sorted(set(sel_params) - {"size"})
        if unexpected:
            raise ValueError(f"Unsupported tournament selection options: {', '.join(unexpected)}")
    tournament_size = (
        int(sel_params.get("size", DEFAULT_TOURNAMENT_PRESSURE)) if sel_method == "tournament" else DEFAULT_TOURNAMENT_PRESSURE
    )
    return sel_method, tournament_size


__all__ = [
    "setup_population",
    "setup_genealogy",
    "setup_selection",
    "DEFAULT_TOURNAMENT_PRESSURE",
]
