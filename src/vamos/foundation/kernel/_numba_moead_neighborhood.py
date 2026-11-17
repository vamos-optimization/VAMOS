"""Compiled MOEA/D neighbourhood replacement.

This module imports numba at import time. MOEA/D reaches it through
:func:`vamos.foundation.kernel.moead_kernels.available_moead_neighborhood_kernels`
(its engine-independent shortcut) and through
``KernelBackend.moead_steady_state_kernels``; ``agg_id`` takes the aggregation
ids of :mod:`vamos.foundation.kernel.moead_kernels`.

**VAMOS 1.4.0:** the update takes the nadir point, the epsilon and
the switch of the normalization of the objectives, and computes the augmented
Tchebycheff (``AGG_AUGMENTED_TCHEBYCHEFF``, the formula of the modified one) and
inverted PBI (``AGG_INVERTED_PBI``) aggregations, and jMetal's modified
Tchebycheff (``AGG_INVERSE_WEIGHT_TCHEBYCHEFF``). Without normalization and
inverted PBI, ``nadir`` is not read; callers pass the ideal point in its place.
"""

from __future__ import annotations

import numpy as np

from .backend import DecisionArray, FloatArray, IndexArray
from .moead_kernels import AGG_INVERSE_WEIGHT_TCHEBYCHEFF, AGG_INVERTED_PBI, AGG_PBI, AGG_TCHEBYCHEFF, AGG_WEIGHTED_SUM
from .numba_ops import njit


@njit(cache=True)
def objective_gap(value: float, ideal: float, nadir: float, epsilon: float, normalize: int, inverted: bool) -> float:
    """The displacement of one objective value that an aggregation measures, as jMetal computes it.

    ``f - z*``, or ``z^nad - f`` for the inverted PBI; with normalization, divided by
    ``z^nad - z* + epsilon``.
    """
    gap = nadir - value if inverted else value - ideal
    if normalize == 1:
        return gap / (nadir - ideal + epsilon)
    return gap


@njit(cache=True)
def update_neighborhood_numba(
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
    replacements = 0
    n_obj = ideal.shape[0]
    for idx in range(candidate_length):
        k = int(candidate_order[idx])

        if agg_id == AGG_TCHEBYCHEFF:
            current_val = -1.0
            child_val = -1.0
            for j in range(n_obj):
                diff_c = abs(objective_gap(F[k, j], ideal[j], nadir[j], agg_epsilon, normalize, False))
                diff_child = abs(objective_gap(child_f[j], ideal[j], nadir[j], agg_epsilon, normalize, False))
                w_eff = weights_safe[k, j]
                val_c = w_eff * diff_c
                val_child = w_eff * diff_child
                if val_c > current_val:
                    current_val = val_c
                if val_child > child_val:
                    child_val = val_child
        elif agg_id == AGG_WEIGHTED_SUM:
            current_val = 0.0
            child_val = 0.0
            for j in range(n_obj):
                w = weights[k, j]
                if normalize == 1:
                    current_val += w * objective_gap(F[k, j], ideal[j], nadir[j], agg_epsilon, 1, False)
                    child_val += w * objective_gap(child_f[j], ideal[j], nadir[j], agg_epsilon, 1, False)
                else:
                    current_val += w * F[k, j]
                    child_val += w * child_f[j]
        elif agg_id == AGG_PBI or agg_id == AGG_INVERTED_PBI:
            inverted = agg_id == AGG_INVERTED_PBI
            d1 = 0.0
            for j in range(n_obj):
                d1 += objective_gap(F[k, j], ideal[j], nadir[j], agg_epsilon, normalize, inverted) * weights_unit[k, j]
            d1 = abs(d1)

            d1_child = 0.0
            for j in range(n_obj):
                d1_child += objective_gap(child_f[j], ideal[j], nadir[j], agg_epsilon, normalize, inverted) * weights_unit[k, j]
            d1_child = abs(d1_child)

            d2 = 0.0
            d2_child = 0.0
            for j in range(n_obj):
                w_unit = weights_unit[k, j]
                diff_c = objective_gap(F[k, j], ideal[j], nadir[j], agg_epsilon, normalize, inverted) - d1 * w_unit
                diff_child = objective_gap(child_f[j], ideal[j], nadir[j], agg_epsilon, normalize, inverted) - d1_child * w_unit
                d2 += diff_c * diff_c
                d2_child += diff_child * diff_child
            d2 = np.sqrt(d2)
            d2_child = np.sqrt(d2_child)
            if inverted:
                # Sato's inverted PBI maximizes d1 - theta d2; jMetal minimizes theta d2 - d1.
                current_val = agg_theta * d2 - d1
                child_val = agg_theta * d2_child - d1_child
            else:
                current_val = d1 + agg_theta * d2
                child_val = d1_child + agg_theta * d2_child
        elif agg_id == AGG_INVERSE_WEIGHT_TCHEBYCHEFF:
            # jMetal's ModifiedTschebyscheff: the largest difference divided by its weight; a zero weight counts 1e-7 times it.
            current_val = -1.0
            child_val = -1.0
            for j in range(n_obj):
                diff_c = abs(objective_gap(F[k, j], ideal[j], nadir[j], agg_epsilon, normalize, False))
                diff_child = abs(objective_gap(child_f[j], ideal[j], nadir[j], agg_epsilon, normalize, False))
                w = weights[k, j]
                if w == 0.0:
                    val_c = 1e-7 * diff_c
                    val_child = 1e-7 * diff_child
                else:
                    val_c = diff_c / w
                    val_child = diff_child / w
                if val_c > current_val:
                    current_val = val_c
                if val_child > child_val:
                    child_val = val_child
        else:
            # Modified and augmented Tchebycheff: the largest weighted difference plus rho times their sum.
            current_val = -1.0
            child_val = -1.0
            sum_c = 0.0
            sum_child = 0.0
            for j in range(n_obj):
                diff_c = abs(objective_gap(F[k, j], ideal[j], nadir[j], agg_epsilon, normalize, False))
                diff_child = abs(objective_gap(child_f[j], ideal[j], nadir[j], agg_epsilon, normalize, False))
                w_eff = weights_safe[k, j]
                val_c = w_eff * diff_c
                val_child = w_eff * diff_child
                if val_c > current_val:
                    current_val = val_c
                if val_child > child_val:
                    child_val = val_child
                sum_c += val_c
                sum_child += val_child
            current_val = current_val + agg_rho * sum_c
            child_val = child_val + agg_rho * sum_child

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


@njit(cache=True)
def update_neighborhood_batch_numba(
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
    for pos in range(children.shape[0]):
        update_neighborhood_numba(
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


__all__ = ["objective_gap", "update_neighborhood_batch_numba", "update_neighborhood_numba"]
