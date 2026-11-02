from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.foundation.constraints.utils import compute_violation

from .neighborhood_kernels import (
    dummy_buffers,
    get_update_neighborhood_batch_numba,
    get_update_neighborhood_numba,
    update_neighborhood_batch_python,
    update_neighborhood_python,
)

if TYPE_CHECKING:
    from .state import MOEADState


def compute_neighbors(weights: np.ndarray[Any, Any], neighbor_size: int) -> np.ndarray[Any, Any]:
    """Compute neighborhood indices based on weight vector distances."""
    dist = np.linalg.norm(weights[:, None, :] - weights[None, :, :], axis=2)
    # Lattice weights have many equidistant neighbours; a stable sort breaks those ties by index, so the
    # neighbourhoods, and the trajectory, are the same on every platform.
    order = np.argsort(dist, axis=1, kind="stable")
    return order[:, :neighbor_size]


def update_neighborhood(
    st: MOEADState,
    idx: int,
    child: np.ndarray[Any, Any],
    child_f: np.ndarray[Any, Any],
    child_g: np.ndarray[Any, Any] | None,
    cv_penalty: float,
    candidate_order: np.ndarray[Any, Any] | None = None,
) -> None:
    """Update neighborhood with a new offspring using aggregation comparison."""
    constraint_mode = st.constraint_mode
    if constraint_mode == "none" or st.G is None or child_g is None:
        constraint_mode = "none"
    use_constraints = constraint_mode != "none"

    if candidate_order is None:
        candidate_order = st.neighbors[idx]
    if candidate_order.size == 0:
        return
    assert st.aggregator is not None
    child_cv = cv_penalty if constraint_mode != "none" else 0.0

    if use_constraints and st.cv is None and st.G is not None:
        st.cv = compute_violation(st.G)

    updater = get_update_neighborhood_numba()
    if updater is None:
        updater = update_neighborhood_python

    dummy_g, dummy_cv, dummy_child_g = dummy_buffers()
    updater(
        st.X,
        st.F,
        st.G if st.G is not None else dummy_g,
        st.cv if st.cv is not None else dummy_cv,
        st.weights,
        st.weights_safe,
        st.weights_unit,
        st.ideal,
        child,
        child_f,
        child_g if child_g is not None else dummy_child_g,
        float(child_cv),
        candidate_order,
        int(candidate_order.shape[0]),
        int(st.replace_limit),
        int(st.aggregation_id),
        float(st.aggregation_theta),
        float(st.aggregation_rho),
        1 if use_constraints else 0,
    )


def update_neighborhood_batch(
    st: MOEADState,
    children: np.ndarray[Any, Any],
    children_f: np.ndarray[Any, Any],
    children_g: np.ndarray[Any, Any] | None,
    children_cv: np.ndarray[Any, Any] | None,
    candidate_orders: np.ndarray[Any, Any],
    candidate_lengths: np.ndarray[Any, Any],
) -> None:
    """Update neighborhoods for a batch of offspring using precomputed candidate orders."""
    if children.shape[0] == 0:
        return

    constraint_mode = st.constraint_mode
    if constraint_mode == "none" or st.G is None or children_g is None:
        constraint_mode = "none"
    use_constraints = constraint_mode != "none"

    if use_constraints and st.cv is None and st.G is not None:
        st.cv = compute_violation(st.G)

    batch_updater = get_update_neighborhood_batch_numba()
    if batch_updater is None:
        batch_updater = update_neighborhood_batch_python

    dummy_g, dummy_cv, dummy_child_g = dummy_buffers()
    if use_constraints:
        if children_cv is None:
            raise ValueError("Constraint-aware MOEA/D batch update requires child violations.")
        child_g_batch = children_g
        child_cv_batch = children_cv
    else:
        child_g_batch = np.empty((children.shape[0], 0), dtype=float)
        child_cv_batch = np.zeros(children.shape[0], dtype=float)

    batch_updater(
        st.X,
        st.F,
        st.G if st.G is not None else dummy_g,
        st.cv if st.cv is not None else dummy_cv,
        st.weights,
        st.weights_safe,
        st.weights_unit,
        st.ideal,
        children,
        children_f,
        child_g_batch if child_g_batch is not None else dummy_child_g.reshape(1, -1),
        child_cv_batch,
        candidate_orders,
        candidate_lengths,
        int(st.replace_limit),
        int(st.aggregation_id),
        float(st.aggregation_theta),
        float(st.aggregation_rho),
        1 if use_constraints else 0,
    )


__all__ = ["compute_neighbors", "update_neighborhood", "update_neighborhood_batch"]
