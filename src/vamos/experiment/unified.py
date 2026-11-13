"""
Unified API for VAMOS optimization.

This module provides a single, powerful entry point that consolidates
problem-based runs, study-style configuration, and auto-parameter defaults
into one flexible function.
"""

from __future__ import annotations

import logging
import numbers
import secrets
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import overload

from vamos.engine.algorithm.catalog import minimum_violation
from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol, AlgorithmName, EngineName
from vamos.experiment.artifacts.specs import RunSpecInputs, build_run_specs, custom_problem_code, is_registry_problem
from vamos.experiment.auto import _compute_max_evaluations, _compute_pop_size, _resolve_problem, _select_algorithm
from vamos.experiment.optimization_result import OptimizationResult, StudyResult
from vamos.experiment.optimize import _build_algorithm_config, _OptimizeConfig, _run_config
from vamos.experiment.runtime.catalog import resolve_engine_details
from vamos.experiment.types import CheckpointPayload, LiveVisualization, TerminationSpec
from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.eval import EvaluationBackend
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.logging import vamos_logging
from vamos.foundation.problem.types import ProblemProtocol
from vamos.foundation.problem.validation import validate_problem


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


#: Default random seed used when no seed is explicitly provided.
DEFAULT_SEED: int = 42


_ALLOWED_EVAL_STRATEGIES = {"serial", "multiprocessing", "dask"}


def _coerce_int(name: str, value: object, *, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ConfigurationError(f"{name} must be an integer.")
    parsed = int(value)
    if min_value is not None and parsed < min_value:
        if min_value == 1:
            raise ConfigurationError(f"{name} must be a positive integer.")
        raise ConfigurationError(f"{name} must be >= {min_value}.")
    return parsed


def _check_algorithm_minimums(algorithm: str, *, pop_size: int | None = None, n_obj: int | None = None, selection: object = None) -> None:
    violation = minimum_violation(algorithm, pop_size=pop_size, n_obj=n_obj, selection=selection)
    if violation is not None:
        raise ConfigurationError(
            violation.message,
            details={"algorithm": algorithm, violation.field: violation.actual, "minimum": violation.minimum},
        )


def _resolve_problem_label(problem: object, instance: ProblemProtocol) -> str:
    if isinstance(problem, str):
        return problem
    for attr in ("name", "key", "label"):
        label = getattr(instance, attr, None)
        if isinstance(label, str) and label.strip():
            return label
    return instance.__class__.__name__


@overload
def optimize(
    problem: str | ProblemProtocol,
    *,
    algorithm: AlgorithmName | str = "auto",
    max_evaluations: int | None = None,
    termination: TerminationSpec | None = None,
    pop_size: int | None = None,
    engine: EngineName | str | None = None,
    seed: int | None = 42,
    verbose: bool = False,
    n_var: int | None = None,
    n_obj: int | None = None,
    problem_kwargs: Mapping[str, object] | None = None,
    algorithm_config: AlgorithmConfigProtocol | None = None,
    eval_strategy: EvaluationBackend | str | None = None,
    live_viz: LiveVisualization | None = None,
    checkpoint: CheckpointPayload | None = None,
) -> OptimizationResult: ...


@overload
def optimize(
    problem: str | ProblemProtocol,
    *,
    algorithm: AlgorithmName | str = "auto",
    max_evaluations: int | None = None,
    termination: TerminationSpec | None = None,
    pop_size: int | None = None,
    engine: EngineName | str | None = None,
    seed: list[int] | tuple[int, ...],
    verbose: bool = False,
    n_var: int | None = None,
    n_obj: int | None = None,
    problem_kwargs: Mapping[str, object] | None = None,
    algorithm_config: AlgorithmConfigProtocol | None = None,
    eval_strategy: EvaluationBackend | str | None = None,
    live_viz: LiveVisualization | None = None,
    checkpoint: CheckpointPayload | None = None,
) -> StudyResult: ...


def optimize(
    problem: str | ProblemProtocol,
    *,
    algorithm: AlgorithmName | str = "auto",
    max_evaluations: int | None = None,
    termination: TerminationSpec | None = None,
    pop_size: int | None = None,
    engine: EngineName | str | None = None,
    seed: int | None | list[int] | tuple[int, ...] = DEFAULT_SEED,
    verbose: bool = False,
    n_var: int | None = None,
    n_obj: int | None = None,
    problem_kwargs: Mapping[str, object] | None = None,
    algorithm_config: AlgorithmConfigProtocol | None = None,
    eval_strategy: EvaluationBackend | str | None = None,
    live_viz: LiveVisualization | None = None,
    checkpoint: CheckpointPayload | None = None,
) -> OptimizationResult | StudyResult:
    """
    Unified entry point for VAMOS optimization.

    This function consolidates multiple APIs into a single powerful interface:
    - Accepts problem names (strings) or instances
    - Quick defaults for ``algorithm="auto"`` and an omitted ``pop_size`` or ``max_evaluations``
    - Handles multi-run studies with seed=[0,1,2,...]
    - Prefer optimize(...) for all runs (explicit options are available).

    The quick defaults suit a first run, not a converged front: 200 individuals
    and 3000 evaluations on ZDT1. The Optimization reference page gives their
    rules. Pass ``pop_size`` and ``max_evaluations`` for an experiment you report.

    Parameters
    ----------
    problem : str | ProblemProtocol
        Problem name (for registered problems) or a problem instance.
    algorithm : AlgorithmName | str, default "auto"
        Algorithm name or ``"auto"`` for automatic selection.
    max_evaluations : int | None, optional
        Maximum function evaluations. Auto-determined when omitted.
    termination : TerminationSpec | None, optional
        Explicit termination pair for advanced runs that also pass
        ``algorithm_config``: ``("max_evaluations", n)``, ``("n_gen", g)``
        (``g * pop_size`` evaluations, the initial population included) or
        ``("hv", {"target_value": t, "reference_point": r, "max_evaluations": n})``.
        Every built-in algorithm accepts all three; counts must be positive integers.
    pop_size : int | None, optional
        Population size. Auto-determined when omitted.
    engine : EngineName | str | None, optional
        Backend engine (for example ``"numpy"``, ``"numba"``, ``"moocore"``,
        or ``"auto"``). Omission keeps the NumPy reference default. ``"auto"``
        prefers loadable Numba for built-in algorithms, otherwise NumPy.
    seed : int | None | list[int] | tuple[int, ...], default ``42``
        Random seed for one run, ``None`` to generate and record a seed before
        execution, or a sequence of explicit seeds for multi-run studies.
    verbose : bool, default ``False``
        Enable VAMOS logging for the run.
    n_var, n_obj : int | None, optional
        Override problem dimensions when using a registered string problem key.
    problem_kwargs : Mapping[str, object] | None, optional
        Extra keyword arguments forwarded to problem instantiation.
    algorithm_config : AlgorithmConfigProtocol | None, optional
        Explicit algorithm config object.
    eval_strategy : EvaluationBackend | str | None, optional
        Evaluation backend name or backend instance.
    live_viz : LiveVisualization | None, optional
        Live visualization callback.
    checkpoint : CheckpointPayload | None, optional
        Warm-start checkpoint (``result.data["checkpoint"]``) for NSGA-II and
        MOEA/D. Resumed with the same configuration and seed, both continue
        exactly as the uninterrupted run, archive constraints included. A
        checkpoint that cannot be restored raises ``ConfigurationError``.
        ``max_evaluations`` is the resumed run's total budget, the checkpoint's
        evaluations included; a smaller budget raises ``ConfigurationError``.
        Multi-seed runs do not accept checkpoints.

    Returns
    -------
    OptimizationResult | StudyResult
        A single-run result for scalar ``seed`` input, or a sequence-compatible
        ``StudyResult`` when ``seed`` is a list/tuple.

    Raises
    ------
    ConfigurationError
        If inputs are invalid or the algorithm/engine combination is not
        supported. A problem object is checked before any evaluation: integer
        ``n_var`` and ``n_obj``, finite ``xl`` and ``xu`` (scalars or arrays of
        shape ``(n_var,)``) with ``xl <= xu``, a known ``encoding`` and a
        callable ``evaluate``.

    Examples:
        # Quick defaults
        >>> result = vamos.optimize("zdt1")

        # Specify algorithm
        >>> result = vamos.optimize("zdt1", algorithm="moead", max_evaluations=5000)

        # Multi-seed study
        >>> study = vamos.optimize("zdt1", seed=[0, 1, 2, 3, 4])
        >>> study.mean("evaluations")
    """
    if isinstance(seed, (list, tuple)):
        if checkpoint is not None:
            raise ConfigurationError("checkpoint is only supported for single-seed runs.")
        if any(isinstance(value, bool) or not isinstance(value, numbers.Integral) for value in seed):
            raise ConfigurationError("seed sequences must contain only integers.")
        with vamos_logging(enabled=verbose):
            return StudyResult(
                [
                    _run_single(
                        problem,
                        algorithm,
                        max_evaluations,
                        termination,
                        pop_size,
                        engine,
                        int(single_seed),
                        verbose,
                        n_var,
                        n_obj,
                        problem_kwargs,
                        algorithm_config,
                        eval_strategy,
                        live_viz,
                        checkpoint,
                    )
                    for single_seed in seed
                ]
            )

    # Single run
    with vamos_logging(enabled=verbose):
        return _run_single(
            problem,
            algorithm,
            max_evaluations,
            termination,
            pop_size,
            engine,
            seed,
            verbose,
            n_var,
            n_obj,
            problem_kwargs,
            algorithm_config,
            eval_strategy,
            live_viz,
            checkpoint,
        )


def _run_single(
    problem: str | ProblemProtocol,
    algorithm: str,
    max_evaluations: int | None,
    termination: TerminationSpec | None,
    pop_size: int | None,
    engine: str | None,
    seed: int | None,
    verbose: bool,
    n_var: int | None,
    n_obj: int | None,
    problem_kwargs: Mapping[str, object] | None,
    algorithm_config: AlgorithmConfigProtocol | None,
    eval_strategy: EvaluationBackend | str | None,
    live_viz: LiveVisualization | None,
    checkpoint: CheckpointPayload | None,
) -> OptimizationResult:
    """Execute a single optimization run."""
    if problem_kwargs is not None and not isinstance(problem_kwargs, Mapping):
        raise ConfigurationError("problem_kwargs must be a mapping of keyword arguments.")
    if n_var is not None:
        n_var = _coerce_int("n_var", n_var, min_value=1)
    if n_obj is not None:
        n_obj = _coerce_int("n_obj", n_obj, min_value=1)
    if pop_size is not None:
        pop_size = _coerce_int("pop_size", pop_size, min_value=1)
    if max_evaluations is not None:
        max_evaluations = _coerce_int("max_evaluations", max_evaluations, min_value=1)
    if termination is not None:
        if algorithm_config is None:
            raise TypeError("optimize() got an unexpected keyword argument 'termination'")
        if not isinstance(termination, tuple) or len(termination) != 2:
            raise ConfigurationError("termination must be a pair such as ('max_evaluations', 10000).")
    requested_seed = seed
    if seed is None:
        seed = secrets.randbits(64)
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral):
        raise ConfigurationError("seed must be an integer.")
    seed = int(seed)
    if isinstance(eval_strategy, str):
        eval_key = eval_strategy.lower()
        if eval_key not in _ALLOWED_EVAL_STRATEGIES:
            choices = ", ".join(sorted(_ALLOWED_EVAL_STRATEGIES))
            raise ConfigurationError(f"eval_strategy must be one of: {choices}.")

    requested_n_var = n_var
    requested_n_obj = n_obj
    requested_pop_size = pop_size
    requested_max_evaluations = max_evaluations
    requested_algorithm = algorithm
    requested_engine = engine

    algorithm_was_auto = algorithm == "auto"

    # Resolve problem
    problem_instance = _resolve_problem(problem, n_var=n_var, n_obj=n_obj, problem_kwargs=problem_kwargs)
    validate_problem(problem_instance)

    # Extract metadata
    n_var = int(problem_instance.n_var)
    n_obj = int(problem_instance.n_obj)
    try:
        encoding = normalize_encoding(getattr(problem_instance, "encoding", "real"))
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc

    if algorithm_config is not None and algorithm == "auto":
        raise ConfigurationError("algorithm_config requires an explicit algorithm name (not algorithm='auto').")

    # Auto-select algorithm if needed
    _auto_defaults: dict[str, str | int] = {}
    if algorithm == "auto":
        algorithm = _select_algorithm(n_obj, encoding)
        _auto_defaults["algorithm"] = algorithm
    # An algorithm_config's population is checked below, with its selection.
    _check_algorithm_minimums(algorithm, pop_size=pop_size if algorithm_config is None else None, n_obj=n_obj)

    # Auto-determine hyperparameters if not specified
    effective_pop_size = pop_size if pop_size else _compute_pop_size(n_var, n_obj)
    effective_max_evaluations = max_evaluations if max_evaluations is not None else _compute_max_evaluations(n_var, n_obj)
    if termination is not None and termination[0] == "max_evaluations":
        effective_max_evaluations = _coerce_int("termination max_evaluations", termination[1], min_value=1)
    if max_evaluations is None and termination is None:
        _auto_defaults["max_evaluations"] = effective_max_evaluations
    effective_termination = termination if termination is not None else ("max_evaluations", effective_max_evaluations)
    effective_engine, engine_source = resolve_engine_details(engine, algorithm=algorithm)
    if engine_source == "auto":
        _auto_defaults["engine"] = effective_engine

    algo_cfg: AlgorithmConfigProtocol
    if algorithm_config is None:
        config_pop_size = effective_pop_size
        if pop_size is None and algorithm.lower() in {"moead", "nsgaiii", "rvea"}:
            config_pop_size = None
        algo_cfg = _build_algorithm_config(
            algorithm,
            pop_size=config_pop_size,
            n_var=n_var,
            n_obj=n_obj,
            encoding=encoding,
        )
    else:
        if not isinstance(algorithm_config, AlgorithmConfigProtocol):
            raise ConfigurationError(
                "algorithm_config must be a config object (e.g., NSGAIIConfig.default(...), or GenericAlgorithmConfig for plugin algorithms)."
            )
        cfg_dict = dict(algorithm_config.to_dict())
        if pop_size is not None:
            cfg_pop_size = cfg_dict.get("pop_size")
            if cfg_pop_size is None:
                raise ConfigurationError(
                    "pop_size cannot be provided unless algorithm_config defines 'pop_size'; set it on algorithm_config instead."
                )
            if isinstance(cfg_pop_size, bool):
                raise ConfigurationError("algorithm_config 'pop_size' must be an int.")
            if isinstance(cfg_pop_size, int):
                cfg_pop_size_int = cfg_pop_size
            elif isinstance(cfg_pop_size, str):
                try:
                    cfg_pop_size_int = int(cfg_pop_size)
                except ValueError as exc:
                    raise ConfigurationError("algorithm_config 'pop_size' must be an int.") from exc
            else:
                raise ConfigurationError("algorithm_config 'pop_size' must be an int.")
            if int(pop_size) != cfg_pop_size_int:
                raise ConfigurationError(
                    f"Conflicting pop_size: pop_size={pop_size} but algorithm_config.pop_size={cfg_pop_size_int}. "
                    "Set pop_size on algorithm_config (single source of truth)."
                )
        algo_cfg = algorithm_config

    algo_cfg_dict = dict(algo_cfg.to_dict())
    resolved_pop_size = algo_cfg_dict.get("pop_size")
    if algorithm_config is not None and isinstance(resolved_pop_size, int) and not isinstance(resolved_pop_size, bool):
        _check_algorithm_minimums(algorithm, pop_size=resolved_pop_size, selection=algo_cfg_dict.get("selection"))
    if resolved_pop_size is None and algorithm_config is None:
        resolved_pop_size = effective_pop_size
    if algorithm_config is None and pop_size is None:
        # The size the run uses: MOEA/D, NSGA-III and RVEA take their reference lattice's.
        _auto_defaults["pop_size"] = resolved_pop_size
    if _auto_defaults:
        _logger().info("[vamos] Auto-selected defaults: %s", _auto_defaults)
    if verbose:
        _logger().info("[vamos] Problem: n_var=%s, n_obj=%s, encoding=%s", n_var, n_obj, encoding)
        _logger().info(
            "[vamos] Config: %s, pop_size=%s, max_evaluations=%s",
            algorithm,
            resolved_pop_size,
            effective_max_evaluations,
        )

    config = _OptimizeConfig(
        problem=problem_instance,
        algorithm=algorithm,
        algorithm_config=algo_cfg,
        termination=effective_termination,
        seed=seed,
        engine=effective_engine,
        eval_strategy=eval_strategy,
        live_viz=live_viz,
        checkpoint=checkpoint,
    )
    started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    started_monotonic = time.perf_counter()
    result = _run_config(config)
    runtime_ms = (time.perf_counter() - started_monotonic) * 1000.0
    completed_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    problem_label = _resolve_problem_label(problem, problem_instance)
    pop_size_source = "config" if algorithm_config is not None else ("explicit" if pop_size is not None else "auto")
    max_evaluations_source = "explicit" if max_evaluations is not None or termination is not None else "auto"
    result.meta["engine_source"] = engine_source
    result.meta["kernel_backend"] = result.meta.get("kernel_backend", effective_engine)
    result.meta["default_sources"] = {
        "algorithm": "auto" if algorithm_was_auto else "explicit",
        "pop_size": pop_size_source,
        "max_evaluations": max_evaluations_source,
        "engine": engine_source,
        "algorithm_config": "auto" if algorithm_config is None else "explicit",
    }
    # Built in only when replay can rebuild it from the registry: never user code under a registry name.
    problem_built_in = is_registry_problem(problem, problem_instance, problem_label)
    requested_spec, resolved_spec = build_run_specs(
        RunSpecInputs(
            problem_built_in=problem_built_in,
            problem_label=problem_label,
            problem_kwargs=problem_kwargs,
            n_var_requested=requested_n_var,
            n_obj_requested=requested_n_obj,
            n_var=n_var,
            n_obj=n_obj,
            encoding=encoding,
            algorithm_requested=requested_algorithm,
            algorithm=algorithm,
            algorithm_config=algo_cfg_dict,
            algorithm_config_explicit=algorithm_config is not None,
            max_evaluations_requested=requested_max_evaluations,
            termination=effective_termination,
            pop_size_requested=requested_pop_size,
            resolved_pop_size=resolved_pop_size,
            engine_requested=requested_engine,
            engine=effective_engine,
            eval_strategy=eval_strategy,
            seed_requested=requested_seed,
            seed=seed,
            default_sources=result.meta["default_sources"],
            problem_code=None if problem_built_in else custom_problem_code(problem_instance),
        )
    )
    result.meta["run_artifact_requested_spec"] = requested_spec
    result.meta["run_artifact_resolved_spec"] = resolved_spec
    result.meta["run_artifact_timestamps"] = {
        "started_at": started_at,
        "completed_at": completed_at,
        "runtime_ms": runtime_ms,
    }
    return result


__all__ = ["optimize"]
