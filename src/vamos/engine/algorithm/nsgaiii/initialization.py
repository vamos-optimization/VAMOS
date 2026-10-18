"""NSGA-III initialization and setup routines.

This module handles algorithm setup including:
- Parsing termination criteria
- Reference direction generation
- Initial population generation
- Archive configuration
- Genealogy tracking setup
"""

from __future__ import annotations

import os
from math import comb
from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.hooks import get_live_viz, setup_genealogy
from vamos.engine.algorithm.components.lifecycle import evaluate_batch, get_eval_strategy
from vamos.engine.algorithm.components.metrics import setup_hv_tracker
from vamos.engine.algorithm.components.population import initialize_population as initialize_decision_population
from vamos.engine.algorithm.components.termination import parse_termination, validate_initial_budget
from vamos.engine.algorithm.components.utils import resolve_bounds_array
from vamos.engine.algorithm.components.weight_vectors import load_or_generate_weight_vectors
from vamos.engine.archive.factory import resolve_external_archive, setup_archive
from vamos.engine.operators.policies.nsgaiii import build_variation_operators
from vamos.foundation.encoding import EncodingLike, normalize_encoding

from .state import NSGAIIIState

if TYPE_CHECKING:
    from vamos.engine.hooks.live_viz import LiveVisualization
    from vamos.foundation.eval.backends import EvaluationBackend
    from vamos.foundation.kernel.backend import KernelBackend
    from vamos.foundation.problem.types import ProblemProtocol


__all__ = [
    "initialize_nsgaiii_run",
    "initialize_population",
]


def initialize_nsgaiii_run(
    config: dict[str, Any],
    kernel: KernelBackend,
    problem: ProblemProtocol,
    termination: tuple[str, Any],
    seed: int,
    eval_strategy: EvaluationBackend | None = None,
    live_viz: LiveVisualization | None = None,
) -> tuple[NSGAIIIState, Any, Any, int, Any]:
    """Initialize NSGA-III run and create state.

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
    max_eval, hv_config = parse_termination(termination, "NSGAIII")

    live_cb = get_live_viz(live_viz)
    eval_strategy = get_eval_strategy(eval_strategy)

    # Setup HV tracker if configured
    hv_tracker = setup_hv_tracker(hv_config, kernel)

    rng = np.random.default_rng(seed)
    pop_size = config["pop_size"]
    encoding = normalize_encoding(getattr(problem, "encoding", "real"))
    xl, xu = resolve_bounds_array(problem, encoding)
    n_var = problem.n_var
    n_obj = problem.n_obj
    constraint_mode = config.get("constraint_mode", "penalty")
    result_mode = config.get("result_mode", "population")

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

    def _handle_refdir_mismatch(expected: int, actual: int, detail: str) -> int:
        raise ValueError(f"{detail} Configure pop_size={expected} or choose reference directions with exactly {actual} points.")

    # Load reference directions (prefer #ref_dirs == pop_size; warn if mismatched)
    dir_cfg = config.get("reference_directions", {}) or {}
    ref_path = dir_cfg.get("path")
    ref_divisions = dir_cfg.get("divisions")
    if ref_path and os.path.exists(ref_path):
        ref_dirs = np.loadtxt(ref_path, delimiter=",")
        ref_dirs = np.atleast_2d(ref_dirs).astype(float, copy=False)
        if ref_dirs.ndim != 2 or ref_dirs.shape[1] != n_obj:
            raise ValueError("Reference directions file has invalid shape.")
        if np.any(ref_dirs < 0.0):
            raise ValueError("Reference directions must be non-negative.")
        row_sums = ref_dirs.sum(axis=1)
        if np.any(np.abs(row_sums - 1.0) > 1e-6):
            raise ValueError("Each reference direction must sum to 1.")
        if ref_dirs.shape[0] != pop_size:
            pop_size = _handle_refdir_mismatch(
                ref_dirs.shape[0],
                pop_size,
                f"NSGA-III reference directions ({ref_dirs.shape[0]}) do not match pop_size ({pop_size}).",
            )
    elif ref_divisions is not None:
        expected = comb(int(ref_divisions) + n_obj - 1, n_obj - 1)
        if pop_size != expected:
            pop_size = _handle_refdir_mismatch(
                expected,
                pop_size,
                f"NSGA-III reference directions for divisions={ref_divisions} expect {expected} points (pop_size={pop_size}).",
            )
        ref_dirs = load_or_generate_weight_vectors(pop_size, n_obj, path=ref_path, divisions=ref_divisions)
    else:
        ref_dirs = load_or_generate_weight_vectors(pop_size, n_obj, path=ref_path, divisions=ref_divisions)
        if ref_dirs.shape[0] != pop_size:
            pop_size = _handle_refdir_mismatch(
                ref_dirs.shape[0],
                pop_size,
                f"NSGA-III reference directions ({ref_dirs.shape[0]}) do not match pop_size ({pop_size}).",
            )

    ref_dirs = np.asarray(ref_dirs, dtype=float)
    validate_initial_budget(max_eval, pop_size, "NSGA-III")
    ref_dirs_norm = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    ref_dirs_norm[np.isnan(ref_dirs_norm)] = 0.0

    # Initialize population
    X, F, G = initialize_population(
        encoding,
        pop_size,
        n_var,
        xl,
        xu,
        rng,
        problem,
        eval_strategy,
        constraint_mode,
        initializer=config.get("initializer"),
    )
    n_eval = pop_size

    # Setup genealogy tracking
    track_genealogy = bool(config.get("track_genealogy", False))
    genealogy_tracker, ids = setup_genealogy(pop_size, F, track_genealogy, "nsgaiii")

    # Setup external archive
    ext_cfg = resolve_external_archive(config)
    archive_X, archive_F, archive_manager = setup_archive(kernel, X, F, n_var, n_obj, X.dtype, ext_cfg, G)
    selection_ranks, selection_crowding = kernel.nsga2_ranking(F)

    # Create state
    state = NSGAIIIState(
        X=X,
        F=F,
        G=G,
        rng=rng,
        pop_size=pop_size,
        offspring_size=pop_size,
        constraint_mode=constraint_mode,
        n_eval=n_eval,
        max_evals=max_eval,
        # NSGA-III specific
        ref_dirs=ref_dirs,
        ref_dirs_norm=ref_dirs_norm,
        pressure=pressure,
        crossover_fn=crossover_fn,
        mutation_fn=mutation_fn,
        ideal_point=np.full(n_obj, np.inf),
        worst_point=np.full(n_obj, -np.inf),
        extreme_points=None,
        selection_ranks=np.asarray(selection_ranks, dtype=int),
        selection_crowding=np.asarray(selection_crowding, dtype=float),
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
        result_mode=result_mode,
    )

    return state, live_cb, eval_strategy, max_eval, hv_tracker


def initialize_population(
    encoding: EncodingLike,
    pop_size: int,
    n_var: int,
    xl: np.ndarray[Any, Any],
    xu: np.ndarray[Any, Any],
    rng: np.random.Generator,
    problem: ProblemProtocol,
    eval_strategy: EvaluationBackend,
    constraint_mode: str,
    initializer: dict[str, Any] | None = None,
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
    normalized = normalize_encoding(encoding)
    X = initialize_decision_population(pop_size, n_var, xl, xu, normalized, rng, problem, initializer=initializer)

    F, G = evaluate_batch(problem, eval_strategy, X, constraint_mode)
    return X, F, G
