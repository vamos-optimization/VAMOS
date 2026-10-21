"""SMS-EMOA initialization and setup routines.

This module handles algorithm setup including:
- Parsing termination criteria
- Initial population generation
- Reference point setup
- Archive configuration
- Genealogy tracking setup
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.hooks import get_live_viz, setup_genealogy
from vamos.engine.algorithm.components.lifecycle import get_eval_strategy
from vamos.engine.algorithm.components.metrics import setup_hv_tracker
from vamos.engine.algorithm.components.population import (
    initialize_population as initialize_decision_population,
)
from vamos.engine.algorithm.components.termination import parse_termination, validate_initial_budget
from vamos.engine.algorithm.components.utils import resolve_bounds_array
from vamos.engine.algorithm.config.base import normalize_constraint_mode
from vamos.engine.archive.factory import resolve_external_archive, setup_archive
from vamos.engine.operators.policies.smsemoa import build_variation_operators
from vamos.foundation.encoding import EncodingLike, normalize_encoding

from .helpers import (
    initialize_reference_point,
    refresh_selection_cache,
)
from .state import SMSEMOAState

if TYPE_CHECKING:
    from vamos.engine.hooks.live_viz import LiveVisualization
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol


__all__ = [
    "initialize_smsemoa_run",
    "initialize_population",
]


def initialize_smsemoa_run(
    config: dict[str, Any],
    kernel: KernelBackend,
    problem: ProblemProtocol,
    termination: tuple[str, Any],
    seed: int,
    eval_strategy: EvaluationBackend | None = None,
    live_viz: LiveVisualization | None = None,
) -> tuple[SMSEMOAState, Any, Any, int, Any]:
    """Initialize SMS-EMOA run and create state.

    Parameters
    ----------
    config : dict
        Algorithm configuration.
    kernel : KernelBackend
        Backend for vectorized operations.
    problem : ProblemProtocol
        Problem to optimize.
    termination : tuple
        Termination criterion, e.g., ("max_evaluations", 10000).
    seed : int
        Random seed for reproducibility.
    eval_strategy : EvaluationBackend, optional
        Evaluation backend for parallel evaluation.
    live_viz : LiveVisualization, optional
        Live visualization callback.

    Returns
    -------
    tuple
        (state, live_cb, eval_strategy, max_eval, hv_tracker)
    """
    max_eval, hv_config = parse_termination(termination, "SMSEMOA")

    live_cb = get_live_viz(live_viz)
    eval_strategy = get_eval_strategy(eval_strategy)

    # Setup HV tracker if configured
    hv_tracker = setup_hv_tracker(hv_config, kernel)

    rng = np.random.default_rng(seed)
    pop_size = config["pop_size"]
    validate_initial_budget(max_eval, int(pop_size), "SMS-EMOA")
    encoding = normalize_encoding(getattr(problem, "encoding", "real"))
    xl, xu = resolve_bounds_array(problem, encoding)
    n_var = problem.n_var
    n_obj = problem.n_obj
    constraint_mode = normalize_constraint_mode(config.get("constraint_mode", "feasibility"))

    # Build variation operators
    crossover_fn, mutation_fn = build_variation_operators(
        config,
        encoding,
        n_var,
        xl,
        xu,
        rng,
        mixed_spec=getattr(problem, "mixed_spec", None),
        variation_kernels=kernel.variation_kernels(),
    )

    # Tournament size
    sel_method, sel_params = config["selection"]
    if sel_method == "tournament":
        unexpected = sorted(set(sel_params) - {"size"})
        if unexpected:
            raise ValueError(f"Unsupported tournament selection options: {', '.join(unexpected)}")
        pressure = int(sel_params.get("size", 2))
    else:
        pressure = 2
    eliminate_duplicates = bool(config.get("eliminate_duplicates", False))

    # Reference point config
    ref_cfg = config.get("reference_point", {}) or {}

    # Initialize population
    X, F, G = initialize_population(
        encoding,
        pop_size,
        n_var,
        xl,
        xu,
        rng,
        problem,
        constraint_mode,
        eval_strategy,
        initializer=config.get("initializer"),
    )
    n_eval = pop_size

    # Setup genealogy tracking
    track_genealogy = bool(config.get("track_genealogy", False))
    genealogy_tracker, ids = setup_genealogy(pop_size, F, track_genealogy, "smsemoa")

    # Initialize reference point
    ref_point, ref_offset, ref_adaptive = initialize_reference_point(F, ref_cfg)

    # Setup external archive
    ext_cfg = resolve_external_archive(config)
    archive_X, archive_F, archive_manager = setup_archive(kernel, X, F, n_var, n_obj, X.dtype, ext_cfg, G)

    # Create state
    state = SMSEMOAState(
        X=X,
        F=F,
        G=G,
        rng=rng,
        pop_size=pop_size,
        offspring_size=1,  # SMS-EMOA typically generates 1 offspring per iteration
        constraint_mode=constraint_mode,
        n_eval=n_eval,
        max_evals=max_eval,
        # SMSEMOA-specific
        ref_point=ref_point,
        ref_offset=ref_offset,
        ref_adaptive=ref_adaptive,
        pressure=pressure,
        selection_method=str(sel_method),
        eliminate_duplicates=eliminate_duplicates,
        crossover_fn=crossover_fn,
        mutation_fn=mutation_fn,
        # Archive
        archive_size=ext_cfg.capacity if ext_cfg else None,
        archive_X=archive_X,
        archive_F=archive_F,
        archive_manager=archive_manager,
        # Termination
        hv_tracker=hv_tracker,
        # Genealogy
        track_genealogy=track_genealogy,
        genealogy_tracker=genealogy_tracker,
        ids=ids,
        result_mode=config.get("result_mode", "non_dominated"),
    )

    refresh_selection_cache(state, kernel)

    return state, live_cb, eval_strategy, max_eval, hv_tracker


def initialize_population(
    encoding: EncodingLike,
    pop_size: int,
    n_var: int,
    xl: np.ndarray[Any, Any],
    xu: np.ndarray[Any, Any],
    rng: np.random.Generator,
    problem: ProblemProtocol,
    constraint_mode: str,
    eval_strategy: EvaluationBackend,
    initializer: Any | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
    """Initialize population based on encoding.

    Parameters
    ----------
    encoding : str
        Variable encoding type.
    pop_size : int
        Population size.
    n_var : int
        Number of variables.
    xl : np.ndarray
        Lower bounds.
    xu : np.ndarray
        Upper bounds.
    rng : np.random.Generator
        Random number generator.
    problem : ProblemProtocol
        Problem for evaluation.
    constraint_mode : str
        Constraint handling mode.

    Returns
    -------
    tuple
        (X, F, G) initial population, objectives, and constraints.
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

    eval_result = eval_strategy.evaluate(X, problem)
    F, G = eval_result.F, eval_result.G
    if constraint_mode == "none":
        G = None
    return X, F, G
