from __future__ import annotations

import logging
import numbers
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any, cast

from vamos.engine.algorithm.config.base import ResultMode
from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol, EngineName
from vamos.engine.algorithm.registry import (
    get_algorithms_registry,
    get_builtin_algorithm_names,
    resolve_algorithm,
    resolve_builtin_algorithm,
)
from vamos.exceptions import ConfigurationError, InvalidAlgorithmError
from vamos.experiment.types import CheckpointPayload, LiveVisualization, TerminationSpec
from vamos.foundation.eval import EvaluationBackend
from vamos.foundation.eval.backends import resolve_eval_strategy
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.foundation.problem.types import ProblemProtocol

if TYPE_CHECKING:
    from .optimization_result import OptimizationResult


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


@dataclass
class _OptimizeConfig:
    """
    Internal configuration container for a single optimization run.
    """

    problem: ProblemProtocol
    algorithm: str
    algorithm_config: AlgorithmConfigProtocol
    termination: TerminationSpec
    seed: int
    engine: EngineName = "numpy"
    eval_strategy: EvaluationBackend | str | None = None  # name or backend instance
    live_viz: LiveVisualization | None = None
    checkpoint: CheckpointPayload | None = None


def _normalize_cfg(cfg: AlgorithmConfigProtocol) -> dict[str, object]:
    return dict(cfg.to_dict())


def _parse_positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ConfigurationError(f"{label} must be an integer.")
    if isinstance(value, numbers.Integral):
        parsed = int(value)
    elif isinstance(value, str):
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ConfigurationError(f"{label} must be an integer.") from exc
    else:
        raise ConfigurationError(f"{label} must be an integer.")
    if parsed <= 0:
        raise ConfigurationError(f"{label} must be a positive integer.")
    return parsed


def _validate_positive_int_field(cfg: dict[str, object], key: str) -> None:
    value = cfg.get(key)
    if value is None:
        return
    label = f"algorithm_config.{key}"
    cfg[key] = _parse_positive_int(value, label=label)


def _validate_algorithm_config(cfg: dict[str, object]) -> None:
    for key in ("pop_size", "offspring_size", "batch_size", "neighbor_size", "replace_limit", "n_partitions"):
        _validate_positive_int_field(cfg, key)


def _termination_budget(termination: TerminationSpec) -> int | None:
    term_type, term_value = termination
    if term_type == "max_evaluations":
        return _parse_positive_int(term_value, label="max_evaluations")
    if term_type == "hv" and isinstance(term_value, dict) and "max_evaluations" in term_value:
        return _parse_positive_int(term_value["max_evaluations"], label="max_evaluations")
    return None


def _validate_budget_covers_population(cfg: dict[str, object], termination: TerminationSpec) -> None:
    pop_raw = cfg.get("pop_size")
    budget = _termination_budget(termination)
    if pop_raw is None or budget is None:
        return
    pop_size = _parse_positive_int(pop_raw, label="algorithm_config.pop_size")
    if budget < pop_size:
        raise ConfigurationError(
            "max_evaluations must be >= pop_size because the initial population consumes pop_size evaluations "
            f"(max_evaluations={budget}, pop_size={pop_size})."
        )


def _with_result_mode(cfg_data: Any, result_mode: ResultMode) -> Any:
    try:
        allowed = {field.name for field in fields(cfg_data)}
    except TypeError:
        return cfg_data
    if "result_mode" not in allowed:
        return cfg_data
    return replace(cfg_data, result_mode=result_mode)


def _run_config(
    config: _OptimizeConfig,
    *,
    engine: EngineName | None = None,
    built_in_only: bool = False,
) -> OptimizationResult:
    """
    Run a single optimization for the provided problem/config pair.

    Parameters
    ----------
    config : _OptimizeConfig
        Internal config with problem, algorithm, and settings.
    engine : EngineName | None, optional
        Override backend engine. If provided, overrides ``config.engine``.

    Returns
    -------
    OptimizationResult
        Pareto-result container with selection helpers and metadata.
    """
    if not isinstance(config, _OptimizeConfig):
        raise ConfigurationError("_run_config() expects an internal optimize config instance.")
    cfg = config

    cfg_dict = _normalize_cfg(cfg.algorithm_config)
    _validate_algorithm_config(cfg_dict)
    _validate_budget_covers_population(cfg_dict, cfg.termination)
    if "engine" in cfg_dict:
        raise ConfigurationError("engine must be configured via optimize(engine=...) rather than algorithm_config.")
    algorithm_raw = cfg.algorithm or ""
    algorithm_name = algorithm_raw.lower()
    available = list(get_builtin_algorithm_names()) if built_in_only else sorted(get_algorithms_registry().keys())
    if not algorithm_name:
        raise ConfigurationError(f"algorithm must be specified. Available: {', '.join(available)}.")
    if algorithm_name not in available:
        raise InvalidAlgorithmError(algorithm_raw, available=available)

    effective_engine = engine or cfg.engine
    kernel = resolve_kernel(effective_engine)

    if cfg.eval_strategy is not None:
        backend = resolve_eval_strategy(cfg.eval_strategy) if isinstance(cfg.eval_strategy, str) else cfg.eval_strategy
    else:
        backend_name = str(cfg_dict.get("eval_strategy", "serial"))
        backend = resolve_eval_strategy(backend_name)

    algo_ctor = resolve_builtin_algorithm(algorithm_name) if built_in_only else resolve_algorithm(algorithm_name)
    algorithm = algo_ctor(cfg_dict, kernel)

    run_fn = cast(Callable[..., dict[str, Any]], algorithm.run)
    if cfg.checkpoint is not None:
        import inspect

        sig = inspect.signature(run_fn)
        if "checkpoint" in sig.parameters or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
            result = run_fn(
                problem=cfg.problem,
                termination=cfg.termination,
                seed=cfg.seed,
                eval_strategy=backend,
                live_viz=cfg.live_viz,
                checkpoint=cfg.checkpoint,
            )
        else:
            raise ConfigurationError(f"Algorithm '{algorithm_name}' does not support checkpoints.")
    else:
        result = run_fn(
            problem=cfg.problem,
            termination=cfg.termination,
            seed=cfg.seed,
            eval_strategy=backend,
            live_viz=cfg.live_viz,
        )
    from .optimization_result import OptimizationResult

    return OptimizationResult(
        result,
        meta={
            "algorithm": algorithm_name,
            "engine": effective_engine,
            "kernel_backend": kernel.name,
            "seed": cfg.seed,
            "termination": cfg.termination,
        },
    )


def _build_algorithm_config(
    algorithm: str,
    *,
    pop_size: int | None,
    n_var: int | None,
    n_obj: int | None,
    encoding: str | None,
) -> AlgorithmConfigProtocol:
    algorithm = algorithm.lower()
    # SMPSO's swarm is not elitist: its result is the leaders archive, as in SMPSOConfig.default().
    result_mode: ResultMode = "non_dominated" if algorithm == "smpso" else "population"

    from vamos.engine.algorithm.config import GenericAlgorithmConfig
    from vamos.engine.algorithm.config.defaults import build_default_algorithm_config

    default_cfg = build_default_algorithm_config(
        algorithm,
        pop_size=pop_size,
        n_var=n_var,
        n_obj=n_obj,
        encoding=encoding,
    )
    if default_cfg is not None:
        return cast(AlgorithmConfigProtocol, _with_result_mode(default_cfg, result_mode))

    registry = get_algorithms_registry()
    if algorithm in registry:
        base: dict[str, object] = {}
        if pop_size is not None:
            base["pop_size"] = pop_size
        if n_var is not None:
            base["n_var"] = n_var
        if n_obj is not None:
            base["n_obj"] = n_obj
        return GenericAlgorithmConfig(base)

    available = sorted(registry.keys())
    raise InvalidAlgorithmError(algorithm, available=available)


__all__: list[str] = []
