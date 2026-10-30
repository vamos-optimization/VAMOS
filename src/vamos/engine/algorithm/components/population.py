from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from vamos.engine.operators.impl.binary import random_binary_population
from vamos.engine.operators.impl.integer import random_integer_population
from vamos.engine.operators.impl.mixed import mixed_initialize
from vamos.engine.operators.impl.permutation import random_permutation_population, validate_permutation_population
from vamos.engine.operators.impl.real import (
    HaltonInitializer,
    LatinHypercubeInitializer,
    OppositionBasedInitializer,
    ScatterSearchInitializer,
    SobolInitializer,
)
from vamos.foundation.encoding import EncodingLike, normalize_encoding

DecisionArray: TypeAlias = NDArray[np.generic]
BoundsArray: TypeAlias = NDArray[np.float64] | NDArray[np.int_]

# Initializers that sample real-valued decision vectors.
_REAL_INITIALIZERS = ("lhs", "scatter", "scatter_search", "sobol", "halton", "obl", "opposition")


def resolve_bounds(problem: Any, encoding: EncodingLike) -> tuple[BoundsArray, BoundsArray]:
    """Return the bounds as arrays of length ``n_var``: integers for integer encoding, floats otherwise."""
    normalized = normalize_encoding(encoding)
    bounds_dtype = int if normalized == "integer" else float
    xl = np.asarray(problem.xl, dtype=bounds_dtype)
    xu = np.asarray(problem.xu, dtype=bounds_dtype)
    n_var = problem.n_var
    if xl.ndim == 0:
        xl = np.full(n_var, xl, dtype=bounds_dtype)
    if xu.ndim == 0:
        xu = np.full(n_var, xu, dtype=bounds_dtype)
    xl = np.ascontiguousarray(xl, dtype=bounds_dtype)
    xu = np.ascontiguousarray(xu, dtype=bounds_dtype)
    return xl, xu


def initialize_population(
    pop_size: int,
    n_var: int,
    xl: DecisionArray,
    xu: DecisionArray,
    encoding: EncodingLike,
    rng: np.random.Generator,
    problem: Any | None = None,
    initializer: Any | None = None,
) -> DecisionArray:
    normalized = normalize_encoding(encoding)
    if pop_size <= 0:
        raise ValueError("pop_size must be positive.")

    def _validate_custom_population(raw: Any, *, label: str) -> DecisionArray:
        arr = np.asarray(raw)
        if arr.ndim != 2 or arr.shape != (pop_size, n_var):
            raise ValueError(f"{label} must return an array of shape ({pop_size}, {n_var}), got {arr.shape}.")
        if normalized == "permutation":
            return validate_permutation_population(arr, label=label)
        if normalized == "binary":
            arr_float = np.asarray(arr, dtype=float)
            if not np.isfinite(arr_float).all() or not np.all((arr_float == 0) | (arr_float == 1)):
                raise ValueError(f"{label} for binary encoding must contain only 0/1 values.")
            return arr_float.astype(np.int8, copy=False)
        if normalized == "integer":
            arr_float = np.asarray(arr, dtype=float)
            if not np.isfinite(arr_float).all() or not np.all(arr_float == np.floor(arr_float)):
                raise ValueError(f"{label} for integer encoding must contain integer-valued genes.")
            arr_int = arr_float.astype(int, copy=False)
            if np.any(arr_int < xl.astype(int)) or np.any(arr_int > xu.astype(int)):
                raise ValueError(f"{label} for integer encoding must respect variable bounds.")
            return arr_int
        if normalized == "mixed":
            return arr.copy()
        return np.asarray(arr, dtype=float)

    if initializer is not None and callable(initializer):
        return _validate_custom_population(initializer(), label="custom initializer")
    if initializer is not None and isinstance(initializer, dict):
        init_type = initializer.get("type", "random").lower()
        if init_type in _REAL_INITIALIZERS and normalized != "real":
            raise ValueError(
                f"Initializer '{init_type}' samples real values and supports only real encoding, not {normalized} encoding. "
                "Use 'random' or a custom initializer."
            )
        if init_type == "custom":
            custom_fn = initializer.get("fn")
            if callable(custom_fn):
                return _validate_custom_population(custom_fn(), label="custom initializer")
            raise ValueError("Custom initializer requires a callable 'fn'.")
        if init_type == "lhs":
            return LatinHypercubeInitializer(pop_size, xl, xu, rng=rng)()
        if init_type in {"scatter", "scatter_search"}:
            base = int(initializer.get("base_size", max(20, pop_size)))
            return ScatterSearchInitializer(pop_size, xl, xu, base_size=base, rng=rng)()
        if init_type == "sobol":
            scramble = bool(initializer.get("scramble", True))
            return SobolInitializer(pop_size, xl, xu, rng=rng, scramble=scramble)()
        if init_type == "halton":
            scramble = bool(initializer.get("scramble", True))
            return HaltonInitializer(pop_size, xl, xu, rng=rng, scramble=scramble)()
        if init_type in {"obl", "opposition"}:
            if problem is None:
                raise ValueError("Opposition-based initializer requires a problem instance for fitness evaluation.")
            return OppositionBasedInitializer(pop_size, xl, xu, problem=problem, rng=rng)()
    if normalized == "permutation":
        return random_permutation_population(pop_size, n_var, rng)
    if normalized == "binary":
        return random_binary_population(pop_size, n_var, rng)
    if normalized == "integer":
        return random_integer_population(pop_size, n_var, xl.astype(int), xu.astype(int), rng)
    if normalized == "mixed":
        spec = getattr(problem, "mixed_spec", None)
        if spec is None:
            raise ValueError("Mixed-encoding problems must define 'mixed_spec'.")
        return mixed_initialize(pop_size, n_var, spec, rng)
    return rng.uniform(np.asarray(xl, dtype=np.float64), np.asarray(xu, dtype=np.float64), size=(pop_size, n_var))


def evaluate_population(problem: Any, X: DecisionArray) -> NDArray[np.float64]:
    out = {"F": np.empty((X.shape[0], problem.n_obj))}
    problem.evaluate(X, out)
    return out["F"]


__all__ = ["resolve_bounds", "initialize_population", "evaluate_population"]
