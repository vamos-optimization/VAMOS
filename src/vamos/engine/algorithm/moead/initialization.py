# algorithm/moead/setup.py
"""
Setup and initialization helpers for MOEA/D.

This module contains functions for parsing configuration, initializing populations,
weight vectors, neighborhoods, and other setup tasks.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from vamos.engine.algorithm.components.hooks import get_live_viz, setup_genealogy
from vamos.engine.algorithm.components.lifecycle import get_eval_strategy
from vamos.engine.algorithm.components.metrics import setup_hv_tracker
from vamos.engine.algorithm.components.population import (
    initialize_population as initialize_decision_population,
)
from vamos.engine.algorithm.components.termination import parse_termination, validate_initial_budget
from vamos.engine.algorithm.components.utils import resolve_bounds_array
from vamos.engine.algorithm.components.weight_vectors import load_or_generate_weight_vectors
from vamos.engine.algorithm.config.base import normalize_constraint_mode
from vamos.engine.archive.factory import resolve_external_archive, setup_archive
from vamos.engine.operators.impl.flags import set_numba_variation
from vamos.engine.operators.policies.moead import build_variation_operators
from vamos.foundation.checkpoint import restore_rng
from vamos.foundation.constraints.utils import compute_violation
from vamos.foundation.encoding import EncodingLike, normalize_encoding
from vamos.foundation.eval.population import evaluate_population_with_constraints

from .helpers import build_aggregator, compute_neighbors, resolve_aggregation_spec
from .state import MOEADState

if TYPE_CHECKING:
    from vamos.engine.algorithm.components.termination import HVTracker
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol
from vamos.engine.hooks.live_viz import LiveVisualization
from vamos.foundation.observer import RunContext


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def initialize_moead_run(
    cfg: dict[str, Any],
    kernel: KernelBackend,
    problem: ProblemProtocol,
    termination: tuple[str, Any],
    seed: int,
    eval_strategy: EvaluationBackend | None = None,
    live_viz: LiveVisualization | None = None,
    checkpoint: Mapping[str, Any] | None = None,
) -> tuple[MOEADState, Any, Any, int, HVTracker]:
    """Initialize MOEA/D state and run-level helpers."""
    max_eval, hv_config = parse_termination(termination, "MOEA/D")

    eval_strategy = get_eval_strategy(eval_strategy)
    live_cb = get_live_viz(live_viz)
    rng = np.random.default_rng(seed)

    pop_size = int(cfg["pop_size"])
    if pop_size < 2:
        raise ValueError("MOEA/D requires pop_size >= 2.")
    validate_initial_budget(max_eval, pop_size, "MOEA/D")

    constraint_mode = normalize_constraint_mode(cfg.get("constraint_mode", "feasibility"))
    encoding = normalize_encoding(getattr(problem, "encoding", "real"))
    xl, xu = resolve_bounds_array(problem, encoding)
    n_var = problem.n_var
    n_obj = problem.n_obj

    X: NDArray[np.generic]
    F: NDArray[np.generic]
    G: NDArray[np.generic] | None
    n_eval: int
    generation = 0
    subproblem_order: NDArray[np.generic]
    subproblem_cursor = 0
    ideal_override: NDArray[np.generic] | None = None
    checkpoint_archive_X: NDArray[np.generic] | None = None
    checkpoint_archive_F: NDArray[np.generic] | None = None

    if checkpoint is not None:
        try:
            X = np.asarray(checkpoint["X"])
            F = np.asarray(checkpoint["F"])
            if X.ndim != 2 or F.ndim != 2:
                raise ValueError("Checkpoint X and F must be 2D arrays.")
            if X.shape[0] != pop_size:
                raise ValueError(f"Checkpoint pop_size={X.shape[0]} does not match config pop_size={pop_size}.")
            if F.shape[0] != X.shape[0]:
                raise ValueError("Checkpoint F row count must match X.")

            G_raw = checkpoint.get("G")
            G = np.asarray(G_raw) if G_raw is not None else None
            if constraint_mode == "none":
                G = None

            n_eval = int(checkpoint.get("n_eval", X.shape[0]))
            if n_eval < X.shape[0]:
                n_eval = X.shape[0]

            rng_state = checkpoint.get("rng_state")
            if rng_state is not None:
                try:
                    restore_rng(rng, cast(dict[str, Any], rng_state))
                except Exception as exc:  # pragma: no cover - defensive
                    _logger().warning("Failed to restore RNG state from checkpoint: %s", exc)

            generation = int(checkpoint.get("generation", 0))
            extra = checkpoint.get("extra", {})
            if isinstance(extra, Mapping):
                order_raw = extra.get("subproblem_order")
                if order_raw is not None:
                    subproblem_order = np.asarray(order_raw, dtype=int)
                    if subproblem_order.ndim != 1 or subproblem_order.shape[0] != pop_size:
                        subproblem_order = rng.permutation(pop_size).astype(int, copy=False)
                else:
                    subproblem_order = rng.permutation(pop_size).astype(int, copy=False)
                subproblem_cursor = int(extra.get("subproblem_cursor", 0))
                if subproblem_cursor < 0 or subproblem_cursor >= pop_size:
                    subproblem_cursor = 0
                ideal_raw = extra.get("ideal")
                if ideal_raw is not None:
                    ideal_override = np.asarray(ideal_raw, dtype=float)
                    if ideal_override.ndim != 1 or ideal_override.shape[0] != n_obj:
                        ideal_override = None
            else:
                subproblem_order = rng.permutation(pop_size).astype(int, copy=False)

            archive_x_raw = checkpoint.get("archive_X")
            archive_f_raw = checkpoint.get("archive_F")
            if archive_x_raw is not None and archive_f_raw is not None:
                checkpoint_archive_X = np.asarray(archive_x_raw)
                checkpoint_archive_F = np.asarray(archive_f_raw)
        except Exception as exc:
            _logger().warning("Invalid checkpoint provided, reinitializing population: %s", exc)
            X, F, G = initialize_population(
                encoding,
                pop_size,
                n_var,
                xl,
                xu,
                rng,
                problem,
                constraint_mode,
                initializer=cfg.get("initializer"),
            )
            n_eval = pop_size
            generation = 0
            subproblem_order = rng.permutation(pop_size).astype(int, copy=False)
            subproblem_cursor = 0
    else:
        X, F, G = initialize_population(
            encoding,
            pop_size,
            n_var,
            xl,
            xu,
            rng,
            problem,
            constraint_mode,
            initializer=cfg.get("initializer"),
        )
        n_eval = pop_size
        subproblem_order = rng.permutation(pop_size).astype(int, copy=False)
        subproblem_cursor = 0

    cv = compute_violation(G) if constraint_mode != "none" and G is not None else None

    weight_cfg = cfg.get("weight_vectors", {}) or {}
    weights = load_or_generate_weight_vectors(
        pop_size,
        n_obj,
        path=weight_cfg.get("path"),
        divisions=weight_cfg.get("divisions"),
        mode="jmetalpy",
    )
    from .helpers import ZERO_WEIGHT_EPS

    weights_safe = np.where(weights == 0, ZERO_WEIGHT_EPS, weights)
    weight_norms = np.linalg.norm(weights, axis=1)
    weight_norms = np.where(weight_norms > 0, weight_norms, 1.0)
    weights_unit = weights / weight_norms[:, None]

    neighbor_size = cfg.get("neighbor_size", min(20, pop_size))
    neighbor_size = max(2, min(neighbor_size, pop_size))
    neighbors = compute_neighbors(weights, neighbor_size)

    aggregation = cfg.get("aggregation", ("pbi", {"theta": 5.0}))
    agg_method, agg_params = aggregation
    aggregator = build_aggregator(agg_method, agg_params)
    agg_id, agg_theta, agg_rho = resolve_aggregation_spec(agg_method, agg_params)

    numba_variation = cfg.get("use_numba_variation")
    if numba_variation is not None:
        set_numba_variation(bool(numba_variation))

    crossover_fn, mutation_fn = build_variation_operators(
        cfg,
        encoding,
        n_var,
        xl,
        xu,
        rng,
        mixed_spec=getattr(problem, "mixed_spec", None),
        variation_kernels=kernel.variation_kernels(),
    )

    ext_cfg = resolve_external_archive(cfg)
    archive_X, archive_F, archive_manager = setup_archive(kernel, X, F, n_var, n_obj, X.dtype, ext_cfg, G)

    hv_tracker = setup_hv_tracker(hv_config, kernel)

    track_genealogy = bool(cfg.get("track_genealogy", False))
    genealogy_tracker, ids = setup_genealogy(pop_size, F, track_genealogy, "moead")

    ctx = RunContext(
        problem=problem,
        algorithm=None,
        config=cfg,
        algorithm_name="moead",
        engine_name=str(kernel.name),
    )
    live_cb.on_start(ctx)

    batch_size = int(cfg.get("batch_size", 1))
    if batch_size <= 0:
        raise ValueError("MOEA/D batch_size must be positive.")
    if batch_size > pop_size:
        batch_size = pop_size

    state = MOEADState(
        X=X,
        F=F,
        G=G,
        rng=rng,
        pop_size=pop_size,
        offspring_size=batch_size,
        constraint_mode=constraint_mode,
        n_eval=n_eval,
        max_evals=max_eval,
        weights=weights,
        weights_safe=weights_safe,
        weights_unit=weights_unit,
        neighbors=neighbors,
        ideal=ideal_override if ideal_override is not None else F.min(axis=0),
        aggregator=aggregator,
        aggregation_id=agg_id,
        aggregation_theta=agg_theta,
        aggregation_rho=agg_rho,
        neighbor_size=neighbor_size,
        delta=float(cfg.get("delta", 0.9)),
        replace_limit=max(1, int(cfg.get("replace_limit", 2))),
        batch_size=batch_size,
        subproblem_order=subproblem_order,
        subproblem_cursor=subproblem_cursor,
        cv=cv,
        crossover_fn=crossover_fn,
        mutation_fn=mutation_fn,
        xl=xl,
        xu=xu,
        archive_size=ext_cfg.capacity if ext_cfg else None,
        archive_X=archive_X,
        archive_F=archive_F,
        archive_manager=archive_manager,
        hv_tracker=hv_tracker,
        track_genealogy=track_genealogy,
        genealogy_tracker=genealogy_tracker,
        ids=ids,
        result_mode=cfg.get("result_mode", "non_dominated"),
    )

    state.generation = generation

    if checkpoint_archive_X is not None and checkpoint_archive_F is not None and ext_cfg:
        try:
            if state.archive_manager is not None:
                state.archive_X, state.archive_F = state.archive_manager.update(checkpoint_archive_X, checkpoint_archive_F)
            else:
                state.archive_X, state.archive_F = checkpoint_archive_X, checkpoint_archive_F
        except Exception as exc:  # pragma: no cover - defensive
            _logger().warning("Failed to restore archive from checkpoint: %s", exc)

    return state, live_cb, eval_strategy, max_eval, hv_tracker


def initialize_population(
    encoding: EncodingLike,
    pop_size: int,
    n_var: int,
    xl: NDArray[np.generic],
    xu: NDArray[np.generic],
    rng: np.random.Generator,
    problem: ProblemProtocol,
    constraint_mode: str,
    initializer: Any | None = None,
) -> tuple[NDArray[np.generic], NDArray[np.generic], NDArray[np.generic] | None]:
    """Initialize population based on encoding.

    Parameters
    ----------
    encoding : str
        Problem encoding: "binary", "integer", "permutation", or "real".
    pop_size : int
        Population size.
    n_var : int
        Number of decision variables.
    xl : NDArray[np.generic]
        Lower bounds.
    xu : NDArray[np.generic]
        Upper bounds.
    rng : np.random.Generator
        Random number generator.
    problem : ProblemProtocol
        The optimization problem.
    constraint_mode : str
        Constraint handling mode.

    Returns
    -------
    tuple[NDArray[np.generic], NDArray[np.generic], NDArray[np.generic] | None]
        (X, F, G) population arrays.
    """
    X = initialize_decision_population(
        pop_size=pop_size,
        n_var=n_var,
        xl=xl,
        xu=xu,
        encoding=encoding,
        rng=rng,
        problem=problem,
        initializer=initializer,
    )

    F, G = evaluate_population_with_constraints(problem, X)
    if constraint_mode == "none":
        G = None
    return X, F, G


__all__ = [
    "initialize_moead_run",
    "initialize_population",
]
