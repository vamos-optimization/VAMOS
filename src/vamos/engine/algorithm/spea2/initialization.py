# algorithm/spea2/setup.py
"""
Setup and initialization helpers for SPEA2.

This module contains functions for parsing configuration, initializing populations,
and setting up archives and trackers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.hooks import get_live_viz, setup_genealogy
from vamos.engine.algorithm.components.lifecycle import get_eval_strategy
from vamos.engine.algorithm.components.metrics import setup_hv_tracker
from vamos.engine.algorithm.components.population import (
    evaluate_population,
    initialize_population,
    resolve_bounds,
)
from vamos.engine.algorithm.components.termination import parse_termination, validate_initial_budget
from vamos.engine.archive.factory import resolve_external_archive, setup_archive
from vamos.engine.operators.policies.spea2 import build_variation_operators
from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.eval.population import evaluate_population_with_constraints

from .helpers import environmental_selection, selection_fitness_density
from .state import SPEA2State

if TYPE_CHECKING:
    from vamos.engine.algorithm.components.termination import HVTracker
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol
from vamos.engine.hooks.live_viz import LiveVisualization
from vamos.foundation.observer import RunContext


def initialize_spea2_run(
    cfg: dict[str, Any],
    kernel: KernelBackend,
    problem: ProblemProtocol,
    termination: tuple[str, Any],
    seed: int,
    eval_strategy: EvaluationBackend | None = None,
    live_viz: LiveVisualization | None = None,
) -> tuple[SPEA2State, Any, Any, int, HVTracker]:
    """Initialize all components for a SPEA2 run.

    Parameters
    ----------
    cfg : dict[str, Any]
        Algorithm configuration.
    kernel : KernelBackend
        Backend for vectorized operations.
    problem : ProblemProtocol
        The optimization problem.
    termination : tuple[str, Any]
        Termination criterion.
    seed : int
        Random seed.
    eval_strategy : EvaluationBackend | None
        Optional evaluation backend.
    live_viz : LiveVisualization | None
        Optional live visualization callback.

    Returns
    -------
    tuple[SPEA2State, Any, Any, int, HVTracker]
        (state, live_cb, eval_strategy, max_eval, hv_tracker)
    """
    max_eval, hv_config = parse_termination(termination, "SPEA2")

    eval_strategy = get_eval_strategy(eval_strategy)
    live_cb = get_live_viz(live_viz)
    rng = np.random.default_rng(seed)

    pop_size = int(cfg.get("pop_size", 100))
    validate_initial_budget(max_eval, pop_size, "SPEA2")
    env_archive_size = int(cfg.get("archive_size", pop_size))
    offspring_size = pop_size
    k_neighbors = cfg.get("k_neighbors")
    k_neighbors = 1 if k_neighbors is None else int(k_neighbors)
    constraint_mode = cfg.get("constraint_mode", "none")

    encoding = normalize_encoding(getattr(problem, "encoding", "real"))
    n_var = problem.n_var
    n_obj = problem.n_obj
    xl, xu = resolve_bounds(problem, encoding)

    # Initialize population
    initializer_cfg = cfg.get("initializer")
    X = initialize_population(pop_size, n_var, xl, xu, encoding, rng, problem, initializer=initializer_cfg)

    if constraint_mode and constraint_mode != "none":
        F, G = evaluate_population_with_constraints(problem, X)
    else:
        F = evaluate_population(problem, X)
        G = None
    n_eval = pop_size

    # Environmental selection for initial internal archive
    env_X, env_F, env_G = environmental_selection(X, F, G, env_archive_size, k_neighbors, constraint_mode)
    selection_raw_fitness, selection_density = selection_fitness_density(env_F, env_G, constraint_mode, k_neighbors)

    # Setup external archive (optional, separate from internal)
    ext_cfg = resolve_external_archive(cfg)
    archive_X, archive_F, archive_manager = setup_archive(
        kernel,
        env_X,
        env_F,
        n_var,
        n_obj,
        X.dtype,
        ext_cfg,
        env_G,
    )

    # Setup HV tracker
    hv_tracker = setup_hv_tracker(hv_config, kernel)

    # Setup genealogy
    track_genealogy = bool(cfg.get("track_genealogy", False))
    genealogy_tracker, ids = setup_genealogy(pop_size, F, track_genealogy, "spea2")

    # Build variation operators
    crossover_fn, mutation_fn = build_variation_operators(cfg, encoding, n_var, xl, xu, rng, problem=problem)

    ctx = RunContext(
        problem=problem,
        algorithm=None,
        config=cfg,
        algorithm_name="spea2",
        engine_name=str(kernel.name),
    )
    live_cb.on_start(ctx)

    # Create state
    state = SPEA2State(
        X=X,
        F=F,
        G=G,
        rng=rng,
        pop_size=pop_size,
        offspring_size=offspring_size,
        constraint_mode=constraint_mode,
        generation=0,
        n_eval=n_eval,
        max_evals=max_eval,
        # External archive (from base class)
        archive_size=ext_cfg.capacity if ext_cfg else None,
        archive_X=archive_X,
        archive_F=archive_F,
        archive_manager=archive_manager,
        result_mode=str(cfg.get("result_mode", "non_dominated")),
        # HV tracking
        hv_tracker=hv_tracker,
        # SPEA2-specific internal archive
        env_X=env_X,
        env_F=env_F,
        env_G=env_G,
        env_archive_size=env_archive_size,
        k_neighbors=k_neighbors,
        crossover_fn=crossover_fn,
        mutation_fn=mutation_fn,
        xl=xl,
        xu=xu,
        selection_raw_fitness=selection_raw_fitness,
        selection_density=selection_density,
        # Genealogy
        track_genealogy=track_genealogy,
        genealogy_tracker=genealogy_tracker,
        ids=ids,
    )

    return state, live_cb, eval_strategy, max_eval, hv_tracker


__all__ = [
    "initialize_spea2_run",
]
