"""Compiled MOEA/D neighbourhood replacement.

This module imports numba at import time. Load it through
:func:`vamos.engine.algorithm.moead.neighborhood_kernels.get_update_neighborhood_numba`,
which keeps numba optional and honours ``VAMOS_USE_NUMBA_MOEAD``.
"""

from __future__ import annotations

import numpy as np

from vamos.foundation.kernel.backend import DecisionArray, FloatArray, IndexArray
from vamos.foundation.kernel.numba_ops import njit

from .aggregation import AGG_PBI, AGG_TCHEBYCHEFF, AGG_WEIGHTED_SUM


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
) -> int:
    replacements = 0
    n_obj = ideal.shape[0]
    for idx in range(candidate_length):
        k = int(candidate_order[idx])

        if agg_id == AGG_TCHEBYCHEFF:
            current_val = -1.0
            child_val = -1.0
            for j in range(n_obj):
                diff_c = abs(F[k, j] - ideal[j])
                diff_child = abs(child_f[j] - ideal[j])
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
                current_val += w * F[k, j]
                child_val += w * child_f[j]
        elif agg_id == AGG_PBI:
            d1 = 0.0
            for j in range(n_obj):
                d1 += (F[k, j] - ideal[j]) * weights_unit[k, j]
            d1 = abs(d1)

            d1_child = 0.0
            for j in range(n_obj):
                d1_child += (child_f[j] - ideal[j]) * weights_unit[k, j]
            d1_child = abs(d1_child)

            d2 = 0.0
            d2_child = 0.0
            for j in range(n_obj):
                w_unit = weights_unit[k, j]
                diff_c = (F[k, j] - ideal[j]) - d1 * w_unit
                diff_child = (child_f[j] - ideal[j]) - d1_child * w_unit
                d2 += diff_c * diff_c
                d2_child += diff_child * diff_child
            d2 = np.sqrt(d2)
            d2_child = np.sqrt(d2_child)
            current_val = d1 + agg_theta * d2
            child_val = d1_child + agg_theta * d2_child
        else:
            current_val = -1.0
            child_val = -1.0
            sum_c = 0.0
            sum_child = 0.0
            for j in range(n_obj):
                diff_c = abs(F[k, j] - ideal[j])
                diff_child = abs(child_f[j] - ideal[j])
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
        )


__all__ = ["update_neighborhood_batch_numba", "update_neighborhood_numba"]
