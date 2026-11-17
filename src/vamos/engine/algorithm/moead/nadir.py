"""The nadir point that MOEA/D's normalized aggregations and its inverted PBI read.

**VAMOS 1.4.0.** This is jMetal 7.8's estimate (``MOEADReplacement`` with
``normalize``): an unbounded ``NonDominatedSolutionListArchive`` receives the initial population
and then each new solution before its replacements, and the nadir point is the largest value of
each objective in that archive (``NadirPoint``, which ignores NaN). The archive compares
solutions with ``DominanceWithConstraintsComparator``: the smaller overall constraint violation
wins, and equal violations (both feasible, say) compare by Pareto dominance; a solution whose
objectives and violation equal a member's is not added. VAMOS keeps the objective vectors and
violations of that archive, the *front*. Unconstrained runs, and runs with
``constraint_mode="none"``, give every solution the violation 0.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.constraints.utils import compute_violation

if TYPE_CHECKING:
    from .state import MOEADState

FloatArray = NDArray[np.float64]


def add_to_front(front_F: FloatArray, front_cv: FloatArray, f: FloatArray, cv: float) -> tuple[FloatArray, FloatArray]:
    """The front after offering it the objective vector ``f`` with violation ``cv``.

    ``f`` joins unless a member dominates it or equals it; the members it dominates leave.
    """
    f = np.asarray(f, dtype=float)
    if front_F.shape[0] == 0:
        return f[None, :].copy(), np.array([float(cv)])
    less_violated = cv < front_cv
    more_violated = cv > front_cv
    same_violation = ~(less_violated | more_violated)
    better = np.any(f < front_F, axis=1)
    worse = np.any(f > front_F, axis=1)
    dominated = more_violated | (same_violation & worse & ~better)
    equal = same_violation & ~better & ~worse
    if bool(np.any(dominated | equal)):
        return front_F, front_cv
    keep = ~(less_violated | (same_violation & better & ~worse))
    return np.vstack([front_F[keep], f[None, :]]), np.append(front_cv[keep], float(cv))


def nondominated_front(F: Any, cv: Any | None) -> tuple[FloatArray, FloatArray]:
    """The front of the rows of ``F`` (violations ``cv``, or 0), offered in order as jMetal's ``addAll``."""
    values = np.asarray(F, dtype=float)
    violations = np.zeros(values.shape[0]) if cv is None else np.asarray(cv, dtype=float)
    front_F: FloatArray = np.empty((0, values.shape[1]))
    front_cv: FloatArray = np.empty(0)
    for row, violation in zip(values, violations, strict=True):
        front_F, front_cv = add_to_front(front_F, front_cv, row, float(violation))
    return front_F, front_cv


def nadir_point(front_F: FloatArray) -> FloatArray:
    """The largest value of each objective in the front; NaN values are ignored, as jMetal's ``NadirPoint`` does."""
    return np.asarray(np.fmax.reduce(front_F, axis=0, initial=-np.inf), dtype=float)


def _violations(st: MOEADState, G: Any | None, n: int) -> FloatArray | None:
    if G is None or st.constraint_mode == "none":
        return None
    return compute_violation(G, n=n)


def start_nadir(st: MOEADState, front_F: Any | None = None, front_cv: Any | None = None) -> None:
    """Start the estimate: the front of a checkpoint when it has one, else that of the population."""
    if front_F is not None:
        values = np.asarray(front_F, dtype=float)
        if values.ndim != 2 or values.shape[1] != st.F.shape[1] or values.shape[0] == 0:
            raise ValueError(f"Checkpoint nadir_front has shape {values.shape}; expected (k, {st.F.shape[1]}) with k > 0.")
        violations = np.zeros(values.shape[0]) if front_cv is None else np.asarray(front_cv, dtype=float)
        if violations.shape != (values.shape[0],):
            raise ValueError(f"Checkpoint nadir_front_cv has shape {violations.shape}; expected ({values.shape[0]},).")
        st.nadir_front, st.nadir_front_cv = values.copy(), violations.copy()
    else:
        st.nadir_front, st.nadir_front_cv = nondominated_front(st.F, _violations(st, st.G, st.F.shape[0]))
    st.nadir = nadir_point(st.nadir_front)


def update_nadir(st: MOEADState, F_children: Any, G_children: Any | None) -> None:
    """Offer the evaluated children to the front, in order, and update the nadir point; nothing without an estimate."""
    if st.nadir_front is None or st.nadir_front_cv is None:
        return
    values = np.asarray(F_children, dtype=float)
    violations = _violations(st, G_children, values.shape[0])
    front_F, front_cv = st.nadir_front, st.nadir_front_cv
    for pos in range(values.shape[0]):
        front_F, front_cv = add_to_front(front_F, front_cv, values[pos], 0.0 if violations is None else float(violations[pos]))
    st.nadir_front, st.nadir_front_cv = front_F, front_cv
    st.nadir = nadir_point(front_F)


__all__ = ["add_to_front", "nadir_point", "nondominated_front", "start_nadir", "update_nadir"]
