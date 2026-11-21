"""NSGA-III initialization and setup routines.

This module handles algorithm setup including:
- Parsing termination criteria
- Reference direction generation
- Initial population generation
- Archive configuration
- Genealogy tracking setup
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.engine.algorithm.components.constraint_handling import run_technique
from vamos.engine.algorithm.components.hooks import get_live_viz, setup_genealogy
from vamos.engine.algorithm.components.lifecycle import evaluate_batch, get_eval_strategy
from vamos.engine.algorithm.components.metrics import setup_hv_tracker
from vamos.engine.algorithm.components.population import initialize_evaluated_population
from vamos.engine.algorithm.components.termination import parse_termination, validate_initial_budget
from vamos.engine.algorithm.components.utils import resolve_bounds_array
from vamos.engine.algorithm.components.weight_vectors import load_or_generate_weight_vectors, vector_source
from vamos.engine.algorithm.config.base import normalize_constraint_mode
from vamos.engine.algorithm.config.nsgaiii import nsgaiii_selection
from vamos.engine.archive.factory import resolve_external_archive, setup_archive
from vamos.engine.operators.policies.nsgaiii import build_variation_operators
from vamos.engine.reference_directions import lattice_size, two_layer_size
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
    max_eval, hv_config = parse_termination(termination, "NSGA-III", pop_size=int(config["pop_size"]))

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
    constraint_mode = normalize_constraint_mode(config.get("constraint_mode", "feasibility"))
    constraint_handling = run_technique(
        config.get("constraint_handling"), constraint_mode, population_size=int(pop_size), max_evaluations=max_eval
    )
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

    # VAMOS 1.5.0: "random" or "tournament" (and its size); any other selection raises.
    selection_method, pressure = nsgaiii_selection(config["selection"])

    # Reference directions: a file, the complete lattice for the configured divisions, or, without
    # either, the complete Das-Dennis lattice with the most directions that fit in pop_size. The
    # population needs at least one individual per direction (N >= H, as Deb and Jain, jMetal and pymoo).
    dir_cfg = vector_source(config.get("reference_directions"), field="reference_directions") or {}
    ref_path = dir_cfg.get("path")
    ref_divisions = dir_cfg.get("divisions")
    ref_inner = dir_cfg.get("inner_divisions")
    if ref_path:
        # VAMOS 1.5.0: the file, or W{n_obj}D_{pop_size}.dat in a directory, is read as MOEA/D
        # reads its weights, with whitespace or commas between the components, and a path that does not
        # exist raises instead of generating a lattice there.
        ref_dirs = load_or_generate_weight_vectors(pop_size, n_obj, path=str(ref_path), sha256=_digest(dir_cfg.get("sha256")))
        source = f"NSGA-III reference directions in '{ref_path}' ({ref_dirs.shape[0]})"
    elif ref_divisions is not None:
        lattice = f"divisions={ref_divisions}" if ref_inner is None else f"divisions={ref_divisions} and inner_divisions={ref_inner}"
        source = f"NSGA-III reference directions for {lattice}"
        if n_obj >= 2:
            # Reject from the binomial count: an oversized lattice can need gigabytes.
            n_directions = (
                lattice_size(n_obj, int(ref_divisions)) if ref_inner is None else two_layer_size(n_obj, int(ref_divisions), int(ref_inner))
            )
            if n_directions > pop_size:
                raise ValueError(
                    f"{source} ({n_directions}) need pop_size >= {n_directions} (pop_size={pop_size}). "
                    "Increase pop_size or choose fewer reference directions."
                )
        ref_dirs = load_or_generate_weight_vectors(
            pop_size, n_obj, divisions=int(ref_divisions), inner_divisions=None if ref_inner is None else int(ref_inner)
        )
        source = f"{source} ({ref_dirs.shape[0]})"
    else:
        if n_obj >= 2 and pop_size < n_obj:
            raise ValueError(
                f"NSGA-III needs pop_size >= n_obj={n_obj} for its smallest reference-direction lattice (pop_size={pop_size})."
            )
        ref_dirs = load_or_generate_weight_vectors(pop_size, n_obj, divisions=None)
        source = f"NSGA-III reference directions ({ref_dirs.shape[0]})"
    if ref_dirs.shape[0] > pop_size:
        raise ValueError(
            f"{source} need pop_size >= {ref_dirs.shape[0]} (pop_size={pop_size}). Increase pop_size or choose fewer reference directions."
        )

    ref_dirs = np.asarray(ref_dirs, dtype=float)
    validate_initial_budget(max_eval, pop_size, "NSGA-III")
    ref_dirs_norm = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    ref_dirs_norm[np.isnan(ref_dirs_norm)] = 0.0

    # Initialize population
    X, F, G, n_eval = initialize_population(
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
        max_evaluations=max_eval,
    )

    # Setup genealogy tracking
    track_genealogy = bool(config.get("track_genealogy", False))
    genealogy_tracker, ids = setup_genealogy(pop_size, F, track_genealogy, "nsgaiii")

    # Setup external archive
    ext_cfg = resolve_external_archive(config, seed=seed)
    archive_X, archive_F, archive_manager = setup_archive(kernel, X, F, n_var, n_obj, X.dtype, ext_cfg, G)

    # Create state
    state = NSGAIIIState(
        X=X,
        F=F,
        G=G,
        rng=rng,
        pop_size=pop_size,
        offspring_size=pop_size,
        constraint_mode=constraint_mode,
        constraint_handling=constraint_handling,
        n_eval=n_eval,
        max_evals=max_eval,
        # NSGA-III specific
        ref_dirs=ref_dirs,
        ref_dirs_norm=ref_dirs_norm,
        selection_method=selection_method,
        pressure=pressure,
        crossover_fn=crossover_fn,
        mutation_fn=mutation_fn,
        ideal_point=np.full(n_obj, np.inf),
        worst_point=np.full(n_obj, -np.inf),
        extreme_points=None,
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
        live_callback_mode=str(config.get("live_callback_mode", "population")),
    )

    return state, live_cb, eval_strategy, max_eval, hv_tracker


def _digest(value: object) -> str | None:
    """The recorded ``sha256`` of the vectors of a file, ``None`` without one."""
    return value if isinstance(value, str) else None


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
    max_evaluations: int | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any] | None, int]:
    """Initialize and evaluate the population based on encoding.

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

    max_evaluations : int | None
        Evaluation budget of the run; the opposition-based initializer needs ``2 * pop_size``.

    Returns
    -------
    tuple
        (X, F, G, n_eval): initial population, objectives, constraints, and every
        evaluation the initialization took.
    """
    return initialize_evaluated_population(
        pop_size,
        n_var,
        xl,
        xu,
        normalize_encoding(encoding),
        rng,
        problem,
        initializer,
        evaluate=lambda X: evaluate_batch(problem, eval_strategy, X, constraint_mode),
        max_evaluations=max_evaluations,
        algorithm="NSGA-III",
    )
