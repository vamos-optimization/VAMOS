"""The push and pull stages of PPS: the equations and algorithms of Fan et al. (2019), section 3.

Each function is one equation or algorithm of the paper, with its number:

- :func:`tchebycheff_values`: Eq. 7, the Tchebycheff function with the reciprocal weights;
- :func:`ideal_and_nadir`: Algorithm 4, the ideal and nadir points of the population;
- :func:`change_rate`: Eqs. 4-6, the largest relative change of the ideal and nadir points over
  the last ``l`` generations;
- :func:`next_epsilon`: Eq. 8, the epsilon level of the pull stage;
- :func:`push_replaces` and :func:`pull_replaces`: Algorithms 1 and 2, whether a new solution
  replaces the solution of a subproblem;
- :func:`feasible_selection`: Algorithm 8, the feasible solutions that the output keeps.

Constraints follow VAMOS's convention, ``G <= 0`` satisfied; the overall violation ``phi`` of
Eq. 3 is the sum of the positive parts of ``G``, so a solution is feasible when ``phi == 0``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.constraints.utils import compute_violation

#: The weight that replaces a zero weight in Eq. 7, whose ``1 / lambda_j`` is not finite for it.
ZERO_WEIGHT = 1e-4
#: ``Delta`` of Eqs. 5 and 6, which keeps their denominators away from zero.
CHANGE_RATE_DELTA = 1e-6
#: The initial values of Algorithm 4: an ideal point above any objective, a nadir point below any.
_UNSET = 1e30


def overall_violation(G: NDArray[Any] | None, n: int) -> NDArray[np.float64]:
    """Eq. 3, ``phi``: the sum of the positive parts of ``G`` per row, ``+inf`` for a row with a NaN constraint.

    A solution whose constraints cannot be evaluated is thus the most violated one: it never
    replaces another in the pull stage, and is replaced by any other.
    """
    phi = compute_violation(G, n=n)
    return np.asarray(np.where(np.isnan(phi), np.inf, phi), dtype=np.float64)


def safe_weights(weights: NDArray[Any]) -> NDArray[np.float64]:
    """The weight vectors with each zero weight replaced by :data:`ZERO_WEIGHT`, so that ``1 / lambda_j`` is finite.

    Eq. 7 divides by the weights; the limit of a weight that tends to zero is a term that
    dominates the others, which a small weight keeps, consistently with the neighbouring vectors.
    """
    values = np.asarray(weights, dtype=np.float64)
    return np.asarray(np.where(values == 0.0, ZERO_WEIGHT, values), dtype=np.float64)


def tchebycheff_values(F: NDArray[Any], weights: NDArray[Any], ideal: NDArray[Any]) -> NDArray[np.float64]:
    """Eq. 7: ``g(x | lambda, z*) = max_j |f_j(x) - z*_j| / lambda_j`` for each row of ``F`` and of ``weights``.

    ``weights`` must have no zero (see :func:`safe_weights`); ``F`` and ``weights`` broadcast row by row.
    """
    return np.asarray(np.max(np.abs(np.asarray(F, dtype=np.float64) - ideal) / weights, axis=-1), dtype=np.float64)


def ideal_and_nadir(F: NDArray[Any]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Algorithm 4: the smallest and largest value of each objective in the population.

    As in the algorithm, the ideal point starts at ``1e30`` and the nadir point at ``-1e30``, and
    a value replaces them only when it is smaller (or larger): NaN values are ignored.
    """
    values = np.asarray(F, dtype=np.float64)
    known = ~np.isnan(values)
    ideal = np.min(values, axis=0, initial=_UNSET, where=known)
    nadir = np.max(values, axis=0, initial=-_UNSET, where=known)
    return np.asarray(ideal, dtype=np.float64), np.asarray(nadir, dtype=np.float64)


def relative_change(current: NDArray[Any], previous: NDArray[Any], delta: float = CHANGE_RATE_DELTA) -> float:
    """Eqs. 5 and 6: ``max_i |current_i - previous_i| / max(|previous_i|, Delta)``."""
    now = np.asarray(current, dtype=np.float64)
    before = np.asarray(previous, dtype=np.float64)
    return float(np.max(np.abs(now - before) / np.maximum(np.abs(before), delta)))


def change_rate(
    ideal: NDArray[Any], previous_ideal: NDArray[Any], nadir: NDArray[Any], previous_nadir: NDArray[Any], delta: float = CHANGE_RATE_DELTA
) -> float:
    """Eq. 4: ``r_k = max(rz_k, rn_k)``, the larger relative change of the ideal point and of the nadir point."""
    return max(relative_change(ideal, previous_ideal, delta), relative_change(nadir, previous_nadir, delta))


def feasible_ratio(phi: NDArray[Any]) -> float:
    """``rf_k``: the ratio of feasible (``phi == 0``) to infeasible solutions of the population, ``inf`` without infeasible ones.

    This is the paper's definition, "the ratio of feasible to infeasible solutions", so with
    ``alpha = 0.95`` epsilon follows its schedule once at least 0.95 / 1.95, about 48.7%, of the
    population is feasible. On LIR-CMOP10, 11 and 12 it agrees with the paper's Table 1, which the
    fraction of feasible solutions, MOEA/D-IEpsilon's ``rf_k``, does not: with the fraction, 5 and
    11 of 30 runs on LIR-CMOP11 and LIR-CMOP12 miss a part of the front, and the mean IGD on
    LIR-CMOP10 is 15% above the paper's (docs/algorithms/pps.md).
    """
    values = np.asarray(phi, dtype=np.float64)
    feasible = int(np.count_nonzero(values == 0.0))
    infeasible = int(values.shape[0]) - feasible
    return math.inf if infeasible == 0 else feasible / infeasible


def control_generation(decay_fraction: float, max_evaluations: int, pop_size: int) -> int:
    """``Tc``: ``decay_fraction`` of the run's generations (the budget over the population, at least 1), rounded half up, at least 1.

    The paper's ``Tc = 800`` with 300,000 evaluations of 300 solutions is ``decay_fraction = 0.8``.
    The rule is that of MOEA/D's ``improved_epsilon`` criterion, by the same authors.
    """
    generations = max(1, int(max_evaluations) // max(1, int(pop_size)))
    return max(1, math.floor(float(decay_fraction) * generations + 0.5))


def next_epsilon(
    epsilon: float, initial_epsilon: float, generation: int, ratio: float, *, tao: float, threshold: float, cp: float, tc: int
) -> float:
    """Eq. 8: the epsilon level of generation ``k`` of the pull stage, ``k < Tc``.

    ``(1 - tao) * epsilon`` when the ratio ``rf_k`` of feasible to infeasible solutions is below ``threshold`` (``alpha``),
    and ``initial_epsilon * (1 - k / Tc) ** cp`` otherwise.
    """
    if ratio < threshold:
        return (1.0 - tao) * epsilon
    return float(initial_epsilon * (1.0 - generation / tc) ** cp)


def push_replaces(child_values: NDArray[Any], values: NDArray[Any]) -> NDArray[np.bool_]:
    """Algorithm 1: the new solution replaces each solution whose Tchebycheff value is not smaller than its own."""
    return np.asarray(np.asarray(child_values) <= np.asarray(values), dtype=np.bool_)


def pull_replaces(
    child_values: NDArray[Any], child_phi: float, values: NDArray[Any], phi: NDArray[Any], epsilon: float
) -> NDArray[np.bool_]:
    """Algorithm 2: whether the new solution replaces each solution in the pull stage.

    When both violations are within ``epsilon``, or they are equal, the Tchebycheff values decide
    (the new solution wins ties); otherwise the less violated solution wins.
    """
    phi = np.asarray(phi, dtype=np.float64)
    by_value = ((child_phi <= epsilon) & (phi <= epsilon)) | (phi == child_phi)
    better = np.asarray(child_values) <= np.asarray(values)
    return np.asarray(np.where(by_value, better, child_phi < phi), dtype=np.bool_)


def first_replacements(replaces: NDArray[Any], limit: int) -> NDArray[np.int64]:
    """The positions of the first ``limit`` solutions replaced, in the order of the candidates (lines 25-34 of Algorithm 3)."""
    return np.asarray(np.flatnonzero(replaces)[:limit], dtype=np.int64)


def distinct_rows(X: NDArray[Any]) -> NDArray[np.int64]:
    """The positions of the first occurrence of each distinct row of ``X``, in their order.

    The union of Algorithm 3's line 37 is a set: a solution of the population that the output
    already holds is counted once.
    """
    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    _, first = np.unique(np.asarray(X), axis=0, return_index=True)
    return np.asarray(np.sort(first), dtype=np.int64)


def feasible_selection(X: NDArray[Any], F: NDArray[Any], phi: NDArray[Any], size: int, kernel: Any) -> NDArray[np.int64]:
    """Algorithm 8: the rows of the distinct feasible solutions of ``X`` that the output keeps, at most ``size``.

    With ``size`` feasible solutions or fewer, all of them; otherwise NSGA-II's selection: the
    first non-dominated fronts, and the solutions of the next one with the largest crowding
    distances (``kernel.nsga2_survival``).
    """
    unique = distinct_rows(X)
    feasible: NDArray[np.int64] = unique[np.asarray(phi, dtype=np.float64)[unique] == 0.0]
    if feasible.size <= size:
        return feasible
    Xf, Ff = np.asarray(X)[feasible], np.asarray(F)[feasible]
    _, _, selected = kernel.nsga2_survival(Xf, Ff, Xf[:0], Ff[:0], size, return_indices=True)
    return np.asarray(feasible[np.asarray(selected, dtype=np.int64)], dtype=np.int64)


__all__ = [
    "CHANGE_RATE_DELTA",
    "ZERO_WEIGHT",
    "change_rate",
    "control_generation",
    "distinct_rows",
    "feasible_ratio",
    "feasible_selection",
    "first_replacements",
    "ideal_and_nadir",
    "next_epsilon",
    "overall_violation",
    "pull_replaces",
    "push_replaces",
    "relative_change",
    "safe_weights",
    "tchebycheff_values",
]
