from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from vamos.engine.algorithm.config.types import AlgorithmConfigProtocol
from vamos.engine.algorithm.variants import canonical_algorithm_name
from vamos.foundation.quality_indicators.hypervolume import hypervolume
from vamos.foundation.quality_indicators.pareto import pareto_filter

_CONSTRAINED_TUNING_UNSUPPORTED = {"agemoea", "rvea"}


class _UnsupportedConstrainedTuningError(RuntimeError):
    """Raised when maintained CLI tuning cannot score constraints safely."""


def _ensure_constrained_tuning_supported(algorithm_name: str, n_constraints: int) -> None:
    """Reject constrained CLI tuning when population-aligned G is unavailable."""
    algo_name = canonical_algorithm_name(algorithm_name)
    if n_constraints > 0 and algo_name in _CONSTRAINED_TUNING_UNSUPPORTED:
        raise _UnsupportedConstrainedTuningError(
            f"Maintained CLI tuning does not yet support constrained {algo_name} runs because the engine does not expose "
            "population-aligned constraint values without changing its stable constraint-mode semantics. "
            "Use an explicit controlled study for this algorithm/problem combination."
        )


def _force_population_result_mode(config: AlgorithmConfigProtocol) -> AlgorithmConfigProtocol:
    """Prefer population result mode for tuning-compatible built-in configs."""
    config_obj: Any = config
    try:
        field_names = {field.name for field in fields(config_obj)}
    except TypeError as exc:
        raise RuntimeError("CLI tuning requires dataclass algorithm configs with result_mode support.") from exc
    if "result_mode" not in field_names:
        raise RuntimeError("CLI tuning algorithm config does not expose result_mode; population-aligned scoring is unavailable.")
    updated: Any = replace(config_obj, result_mode="population")
    return cast(AlgorithmConfigProtocol, updated)


def _population_front_for_scoring(result: Any) -> NDArray[np.float64]:
    payload = getattr(result, "data", None)
    if not isinstance(payload, dict):
        raise RuntimeError("Tuning evaluator requires OptimizationResult.data with a final population payload.")
    population = payload.get("population")
    if not isinstance(population, Mapping):
        raise RuntimeError("Tuning evaluator requires result.data['population'] for source-consistent scoring.")
    raw_f = population.get("F")
    if raw_f is None:
        raise RuntimeError("Tuning evaluator final population does not contain objective values 'F'.")

    F = np.asarray(raw_f, dtype=float)
    if F.ndim != 2:
        raise RuntimeError(f"Tuning evaluator expected population F to be 2-D, got shape {F.shape}.")

    raw_g = population.get("G")
    constraint_source = "population"
    if raw_g is None:
        try:
            n_constraints = int(payload.get("_tuning_n_constraints", 0))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Tuning evaluator received invalid constraint-count metadata.") from exc

        if n_constraints > 0:
            raw_top_f = payload.get("F")
            if raw_top_f is None:
                raise RuntimeError("Constrained tuning requires population-aligned constraint values G.")
            top_f = np.asarray(raw_top_f, dtype=float)
            if top_f.shape != F.shape or not np.array_equal(top_f, F, equal_nan=True):
                raise RuntimeError(
                    "Constrained tuning requires top-level F to align with population F when population G is unavailable."
                )
            raw_g = payload.get("G")
            constraint_source = "top-level"
            if raw_g is None:
                raise RuntimeError("Constrained tuning result does not contain population-aligned constraint values G.")

    if raw_g is not None:
        G = np.asarray(raw_g, dtype=float)
        if G.ndim == 1:
            G = G[:, None]
        if G.ndim != 2 or G.shape[0] != F.shape[0]:
            raise RuntimeError(f"Tuning evaluator {constraint_source} G must align row-wise with population F.")
        F = F[np.all(G <= 0.0, axis=1)]

    if len(F) == 0:
        return np.empty((0, F.shape[1]), dtype=float)
    front = pareto_filter(F, return_indices=False)
    if front is None:
        return np.empty((0, F.shape[1]), dtype=float)
    return np.asarray(front, dtype=float)


def _score_population_result(
    result: Any,
    ref_point: list[float],
    runtime_penalty: float,
    failure_score: float,
) -> float:
    """Score one tuning result while distinguishing valid empty fronts from failures."""
    payload = getattr(result, "data", None)
    failed = isinstance(payload, dict) and bool(payload.get("_tuning_failed", False))
    if failed:
        base_hv = float(failure_score)
    else:
        F = _population_front_for_scoring(result)
        ref = np.asarray(ref_point, dtype=float)
        contributing = F[np.all(F <= ref, axis=1)]
        base_hv = float(hypervolume(contributing, ref)) if len(contributing) > 0 else 0.0

    elapsed_s = 0.0
    if isinstance(payload, dict):
        elapsed_raw = payload.get("_elapsed_s", 0.0)
        try:
            elapsed_s = float(elapsed_raw)
        except Exception:
            elapsed_s = 0.0
    return float(base_hv - float(runtime_penalty) * float(np.log1p(max(0.0, elapsed_s))))
