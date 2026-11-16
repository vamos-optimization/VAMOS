from __future__ import annotations

from collections.abc import Callable

import numpy as np

from vamos.foundation.kernel.backend import DecisionArray, FloatArray, IndexArray
from vamos.foundation.kernel.moead_kernels import available_moead_neighborhood_kernels

from .aggregation import AGG_INVERTED_PBI, AGG_PBI, AGG_TCHEBYCHEFF, AGG_WEIGHTED_SUM

_UPDATE_NEIGHBORHOOD_JIT: Callable[..., int] | None = None
_UPDATE_NEIGHBORHOOD_BATCH_JIT: Callable[..., None] | None = None
_UPDATE_NEIGHBORHOOD_DISABLED = False
_DUMMY_G: FloatArray | None = None
_DUMMY_CV: FloatArray | None = None
_DUMMY_CHILD_G: FloatArray | None = None


def dummy_buffers() -> tuple[FloatArray, FloatArray, FloatArray]:
    global _DUMMY_G, _DUMMY_CV, _DUMMY_CHILD_G
    if _DUMMY_G is None:
        _DUMMY_G = np.empty((0, 0), dtype=float)
    if _DUMMY_CV is None:
        _DUMMY_CV = np.empty(0, dtype=float)
    if _DUMMY_CHILD_G is None:
        _DUMMY_CHILD_G = np.empty(0, dtype=float)
    return _DUMMY_G, _DUMMY_CV, _DUMMY_CHILD_G


def _use_numba_moead() -> bool:
    import os

    flag = os.environ.get("VAMOS_USE_NUMBA_MOEAD")
    if flag is None or flag == "":
        return True
    return flag.lower() in {"1", "true", "yes", "on"}


def get_update_neighborhood_numba() -> Callable[..., int] | None:
    global _UPDATE_NEIGHBORHOOD_BATCH_JIT, _UPDATE_NEIGHBORHOOD_DISABLED, _UPDATE_NEIGHBORHOOD_JIT
    if _UPDATE_NEIGHBORHOOD_DISABLED:
        return None
    if _UPDATE_NEIGHBORHOOD_JIT is not None:
        return _UPDATE_NEIGHBORHOOD_JIT
    if not _use_numba_moead():
        _UPDATE_NEIGHBORHOOD_DISABLED = True
        return None
    compiled = available_moead_neighborhood_kernels()
    if compiled is None:
        _UPDATE_NEIGHBORHOOD_DISABLED = True
        return None
    _UPDATE_NEIGHBORHOOD_JIT, _UPDATE_NEIGHBORHOOD_BATCH_JIT = compiled
    return _UPDATE_NEIGHBORHOOD_JIT


def get_update_neighborhood_batch_numba() -> Callable[..., None] | None:
    if _UPDATE_NEIGHBORHOOD_DISABLED:
        return None
    if _UPDATE_NEIGHBORHOOD_BATCH_JIT is not None:
        return _UPDATE_NEIGHBORHOOD_BATCH_JIT
    if get_update_neighborhood_numba() is None:
        return None
    return _UPDATE_NEIGHBORHOOD_BATCH_JIT


def _gap(value: float, ideal: float, nadir: float, epsilon: float, normalize: int, inverted: bool) -> float:
    # The compiled objective_gap, operation for operation.
    gap = nadir - value if inverted else value - ideal
    return gap / (nadir - ideal + epsilon) if normalize == 1 else gap


def _scalar_aggregation(
    f: FloatArray,
    k: int,
    weights: FloatArray,
    weights_safe: FloatArray,
    weights_unit: FloatArray,
    ideal: FloatArray,
    nadir: FloatArray,
    agg_id: int,
    agg_theta: float,
    agg_rho: float,
    agg_epsilon: float,
    normalize: int,
) -> float:
    """The aggregation value of ``f`` in subproblem ``k`` with the loops of the compiled update.

    Used for the inverted PBI and the normalized objectives, so that this fallback gives the
    compiled update's values bit for bit.
    """
    n_obj = ideal.shape[0]
    if agg_id == AGG_WEIGHTED_SUM:
        total = 0.0
        for j in range(n_obj):
            total += float(weights[k, j]) * _gap(float(f[j]), float(ideal[j]), float(nadir[j]), agg_epsilon, 1, False)
        return total
    if agg_id in {AGG_PBI, AGG_INVERTED_PBI}:
        inverted = agg_id == AGG_INVERTED_PBI
        gaps = [_gap(float(f[j]), float(ideal[j]), float(nadir[j]), agg_epsilon, normalize, inverted) for j in range(n_obj)]
        d1 = 0.0
        for j in range(n_obj):
            d1 += gaps[j] * float(weights_unit[k, j])
        d1 = abs(d1)
        d2 = 0.0
        for j in range(n_obj):
            diff = gaps[j] - d1 * float(weights_unit[k, j])
            d2 += diff * diff
        d2 = float(np.sqrt(d2))
        return agg_theta * d2 - d1 if inverted else d1 + agg_theta * d2
    largest = -1.0
    total = 0.0
    for j in range(n_obj):
        value = float(weights_safe[k, j]) * abs(_gap(float(f[j]), float(ideal[j]), float(nadir[j]), agg_epsilon, normalize, False))
        largest = max(largest, value)
        total += value
    return largest if agg_id == AGG_TCHEBYCHEFF else largest + agg_rho * total


def update_neighborhood_python(
    X: DecisionArray,
    F: FloatArray,
    G: FloatArray,
    cv: FloatArray,
    weights: FloatArray,
    weights_safe: FloatArray,
    weights_unit: FloatArray,
    ideal: FloatArray,
    child: DecisionArray,
    child_f: FloatArray,
    child_g: FloatArray,
    child_cv: float,
    candidate_order: IndexArray,
    candidate_length: int,
    replace_limit: int,
    agg_id: int,
    agg_theta: float,
    agg_rho: float,
    use_constraints: int,
    nadir: FloatArray,
    agg_epsilon: float,
    normalize: int,
) -> int:
    """Pure-Python fallback for neighborhood updates when numba is unavailable.

    ``nadir``, ``agg_epsilon`` and ``normalize`` are read only with normalization or the inverted PBI.
    """
    replacements = 0
    for idx in range(candidate_length):
        k = int(candidate_order[idx])

        diff_current = np.abs(F[k] - ideal)
        diff_child = np.abs(child_f - ideal)

        if normalize == 1 or agg_id == AGG_INVERTED_PBI:
            args = (k, weights, weights_safe, weights_unit, ideal, nadir, agg_id, agg_theta, agg_rho, agg_epsilon, normalize)
            current_val = _scalar_aggregation(F[k], *args)
            child_val = _scalar_aggregation(child_f, *args)
        elif agg_id == AGG_TCHEBYCHEFF:
            current_val = float(np.max(weights_safe[k] * diff_current))
            child_val = float(np.max(weights_safe[k] * diff_child))
        elif agg_id == AGG_WEIGHTED_SUM:
            current_val = float(np.dot(weights[k], F[k]))
            child_val = float(np.dot(weights[k], child_f))
        elif agg_id == AGG_PBI:
            d1 = abs(float(np.dot(F[k] - ideal, weights_unit[k])))
            d1_child = abs(float(np.dot(child_f - ideal, weights_unit[k])))
            d2 = float(np.linalg.norm((F[k] - ideal) - d1 * weights_unit[k]))
            d2_child = float(np.linalg.norm((child_f - ideal) - d1_child * weights_unit[k]))
            current_val = d1 + agg_theta * d2
            child_val = d1_child + agg_theta * d2_child
        else:
            weighted_current = weights_safe[k] * diff_current
            weighted_child = weights_safe[k] * diff_child
            current_val = float(np.max(weighted_current)) + agg_rho * float(np.sum(weighted_current))
            child_val = float(np.max(weighted_child)) + agg_rho * float(np.sum(weighted_child))

        replace = False
        if use_constraints == 1:
            current_cv = cv[k]
            feas_child = child_cv <= 0.0
            feas_curr = current_cv <= 0.0
            if (not feas_curr) and feas_child:
                replace = True
            elif feas_child and feas_curr:
                replace = child_val < current_val
            else:
                replace = child_cv < current_cv
        else:
            replace = child_val < current_val

        if not replace:
            continue

        X[k] = child
        F[k] = child_f
        if use_constraints == 1:
            G[k] = child_g
            cv[k] = child_cv
        replacements += 1
        if replacements >= replace_limit:
            break
    return replacements


def update_neighborhood_batch_python(
    X: DecisionArray,
    F: FloatArray,
    G: FloatArray,
    cv: FloatArray,
    weights: FloatArray,
    weights_safe: FloatArray,
    weights_unit: FloatArray,
    ideal: FloatArray,
    children: DecisionArray,
    children_f: FloatArray,
    children_g: FloatArray,
    children_cv: FloatArray,
    candidate_orders: IndexArray,
    candidate_lengths: IndexArray,
    replace_limit: int,
    agg_id: int,
    agg_theta: float,
    agg_rho: float,
    use_constraints: int,
    nadir: FloatArray,
    agg_epsilon: float,
    normalize: int,
) -> None:
    """Pure-Python batch fallback mirroring the JIT neighborhood update semantics."""
    for pos in range(children.shape[0]):
        update_neighborhood_python(
            X,
            F,
            G,
            cv,
            weights,
            weights_safe,
            weights_unit,
            ideal,
            children[pos],
            children_f[pos],
            children_g[pos],
            float(children_cv[pos]),
            candidate_orders[pos],
            int(candidate_lengths[pos]),
            replace_limit,
            agg_id,
            agg_theta,
            agg_rho,
            use_constraints,
            nadir,
            agg_epsilon,
            normalize,
        )


__all__ = [
    "dummy_buffers",
    "get_update_neighborhood_batch_numba",
    "get_update_neighborhood_numba",
    "update_neighborhood_batch_python",
    "update_neighborhood_python",
]
