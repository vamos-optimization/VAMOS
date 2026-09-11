"""SMPSO initialization and setup routines.

This module handles algorithm setup including:
- Population initialization
- Velocity initialization
- Leader archive setup
- Genealogy tracking setup
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.archive import CrowdingDistanceArchive
from vamos.engine.algorithm.components.hooks import get_live_viz, setup_genealogy
from vamos.engine.algorithm.components.lifecycle import get_eval_strategy
from vamos.engine.algorithm.components.metrics import setup_hv_tracker
from vamos.engine.algorithm.components.population import (
    initialize_population,
    resolve_bounds,
)
from vamos.engine.algorithm.components.termination import parse_termination, validate_initial_budget
from vamos.engine.operators.policies.smpso import (
    build_mutation_operator,
    build_repair_operator,
)
from vamos.foundation.encoding import normalize_encoding

from .state import SMPSOState

if TYPE_CHECKING:
    from vamos.engine.hooks.live_viz import LiveVisualization
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol


__all__ = [
    "initialize_smpso_run",
]


def initialize_smpso_run(
    config: dict[str, Any],
    kernel: KernelBackend,
    problem: ProblemProtocol,
    termination: tuple[str, Any],
    seed: int,
    eval_strategy: EvaluationBackend | None = None,
    live_viz: LiveVisualization | None = None,
) -> tuple[SMPSOState, Any, Any, int, Any]:
    """Initialize SMPSO run and create state.

    Parameters
    ----------
    config : dict
        Algorithm configuration.
    kernel : KernelBackend
        Backend for vectorized operations.
    problem : ProblemProtocol
        Problem to optimize.
    termination : tuple
        Termination criterion.
    seed : int
        Random seed.
    eval_strategy : EvaluationBackend, optional
        Evaluation backend.
    live_viz : LiveVisualization, optional
        Visualization callback.

    Returns
    -------
    tuple
        (state, live_cb, eval_strategy, max_eval, hv_tracker)
    """
    max_eval, hv_config = parse_termination(termination, "SMPSO")
    eval_strategy = get_eval_strategy(eval_strategy)
    live_cb = get_live_viz(live_viz)
    rng = np.random.default_rng(seed)

    pop_size = int(config.get("pop_size", 100))
    validate_initial_budget(max_eval, pop_size, "SMPSO")
    archive_size = int(config.get("archive_size", pop_size))
    inertia = float(config.get("inertia", 0.1))
    c1 = float(config.get("c1", 1.5))
    c2 = float(config.get("c2", 1.5))
    vmax_fraction = config.get("vmax_fraction")
    if vmax_fraction is not None:
        vmax_fraction = float(vmax_fraction)

    c1_min = float(config.get("c1_min", c1))
    c1_max = float(config.get("c1_max", c1_min + 1.0))
    c2_min = float(config.get("c2_min", c2))
    c2_max = float(config.get("c2_max", c2_min + 1.0))
    r1_min = float(config.get("r1_min", 0.0))
    r1_max = float(config.get("r1_max", 1.0))
    r2_min = float(config.get("r2_min", 0.0))
    r2_max = float(config.get("r2_max", 1.0))
    min_weight = float(config.get("min_weight", inertia))
    max_weight = float(config.get("max_weight", min_weight))
    change_velocity1 = float(config.get("change_velocity1", -1.0))
    change_velocity2 = float(config.get("change_velocity2", -1.0))
    mutation_every = int(config.get("mutation_every", 6))

    encoding = normalize_encoding(getattr(problem, "encoding", "real"))
    if encoding not in {"real", "mixed"}:
        raise ValueError(f"SMPSO does not support encoding '{encoding}'.")

    n_var = int(problem.n_var)
    n_obj = int(problem.n_obj)
    xl, xu = resolve_bounds(problem, encoding)

    span = xu - xl
    if vmax_fraction is None:
        delta_max = np.abs(span) / 2.0
    else:
        delta_max = np.abs(span) * vmax_fraction
    delta_min = -delta_max

    # Initialize population
    initializer_cfg = config.get("initializer")
    X = initialize_population(pop_size, n_var, xl, xu, encoding, rng, problem, initializer=initializer_cfg)

    # Evaluate initial population
    eval_init = eval_strategy.evaluate(X, problem)
    F = eval_init.F
    constraint_mode = config.get("constraint_mode", "feasibility")
    G = eval_init.G if constraint_mode != "none" else None

    # Initialize velocity (jMetalPy starts at zero velocity)
    velocity = np.zeros_like(X, dtype=float)

    # Personal bests
    pbest_X = X.copy()
    pbest_F = F.copy()
    pbest_G = G.copy() if G is not None else None

    # Leader archive
    leader_archive = CrowdingDistanceArchive(archive_size, n_var, n_obj, X.dtype)
    archive_X, archive_F = leader_archive.update(X, F, G)
    archive_crowding = None
    if archive_F is not None and archive_F.size:
        from vamos.engine.algorithm.components.subset_selection import _single_front_crowding

        archive_crowding = _single_front_crowding(archive_F)

    # Genealogy tracking
    track_genealogy = bool(config.get("track_genealogy", False))
    genealogy_tracker, ids = setup_genealogy(pop_size, F, track_genealogy, "smpso")

    # Build operators
    mutation_op = build_mutation_operator(config, encoding, n_var, xl, xu, problem=problem)
    repair_op = build_repair_operator(config, encoding)

    # HV tracker
    hv_tracker = setup_hv_tracker(hv_config, kernel)

    # Create state
    state = SMPSOState(
        X=X,
        F=F,
        G=G,
        rng=rng,
        pop_size=pop_size,
        offspring_size=pop_size,
        constraint_mode=constraint_mode,
        n_eval=pop_size,
        max_evals=max_eval,
        generation=0,
        # Archive (leaders)
        archive_size=archive_size,
        archive_X=archive_X,
        archive_F=archive_F,
        archive_manager=leader_archive,
        # Termination
        hv_tracker=hv_tracker,
        # PSO state
        velocity=velocity,
        pbest_X=pbest_X,
        pbest_F=pbest_F,
        pbest_G=pbest_G,
        inertia=inertia,
        c1=c1,
        c2=c2,
        c1_min=c1_min,
        c1_max=c1_max,
        c2_min=c2_min,
        c2_max=c2_max,
        r1_min=r1_min,
        r1_max=r1_max,
        r2_min=r2_min,
        r2_max=r2_max,
        min_weight=min_weight,
        max_weight=max_weight,
        change_velocity1=change_velocity1,
        change_velocity2=change_velocity2,
        mutation_every=mutation_every,
        vmax=delta_max,
        delta_max=delta_max,
        delta_min=delta_min,
        xl=xl,
        xu=xu,
        mutation_op=mutation_op,
        repair_op=repair_op,
        archive_crowding=archive_crowding,
        # Genealogy
        track_genealogy=track_genealogy,
        genealogy_tracker=genealogy_tracker,
        ids=ids,
    )

    return state, live_cb, eval_strategy, max_eval, hv_tracker
