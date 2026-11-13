"""Phase 1 of ToP and the switch to phase 2.

Phase 1 of ToP (Liu and Wang, 2019) solves the problem as a constrained single-objective one,
whose objective is the mean of the objectives, with a one-to-one replacement by the feasibility
rule. It lasts while ``(delta >= 0.2 or Pf <= 1/3)`` and the evaluations done are not more than
90% of the maximum, where ``Pf`` is the ratio of feasible solutions of the population and
``delta`` measures how close the best third of the feasible solutions are in the normalized
objective space. These functions transcribe jMetal's port (pull request jMetal/jMetal#525, class
``ToP``), which follows PlatEMO's implementation where it differs from the paper:

- the overall violation is the sum of the violated parts of the constraints, and a solution is
  feasible when it is zero;
- the mean of the objectives replaces their sum (the same comparisons), and with equal violations
  the trial vector replaces its parent only if its mean objective is strictly smaller, also when
  both are infeasible;
- ``delta`` normalizes the objectives with the minimum and maximum of the feasible solutions of
  the current population, keeps its value while fewer than three solutions are feasible, and is
  ``NaN`` when an objective has a single value among them;
- the best solution, whose position the DE/rand-to-best/1/bin operator uses, is the first one with
  the lowest mean objective, feasible or not.

Every sum of objectives adds its terms in order from 0.0, as jMetal's loops, so that the
floating-point values, and the ties they create, are jMetal's.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.constraints.utils import compute_violation

FloatArray = NDArray[np.float64]

# Phase 1 goes on while delta is at least this value (or Pf is not above the next one).
DELTA_THRESHOLD = 0.2
# Phase 1 goes on while Pf is not above this value (or delta is at least the previous one).
FEASIBLE_RATIO_THRESHOLD = 1.0 / 3.0
# Phase 1 ends when the evaluations done exceed this fraction of the maximum.
PHASE_ONE_EVALUATION_FRACTION = 0.9
# Probability of DE/current-to-rand/1 (otherwise, DE/rand-to-best/1/bin) for each trial vector of phase 1.
CURRENT_TO_RAND_PROBABILITY = 0.5
# Value of delta at the beginning of the run.
INITIAL_DELTA = 1.0


def constraint_violation(G: NDArray[Any] | None, n: int) -> FloatArray:
    """Overall violation of each row: VAMOS's :func:`~vamos.foundation.constraints.utils.compute_violation`, the sum of the positive constraint values.

    VAMOS's ``g <= 0`` is feasible, so this is jMetal's ``|overallConstraintViolationDegree|`` of
    the negated values. jMetal adds the violated parts with ``DoubleStream.sum``, a compensated
    sum, so with two or more violated constraints the two can differ in the last bit. Without
    constraint values every row has violation zero.
    """
    return compute_violation(G, n=int(n))


def mean_objectives(F: NDArray[Any]) -> FloatArray:
    """Mean of the objectives of each row, the objective of phase 1: their sum from 0.0 divided by their number."""
    objectives = np.asarray(F, dtype=float)
    total = np.zeros(objectives.shape[0], dtype=float)
    for j in range(objectives.shape[1]):
        total = total + objectives[:, j]
    return np.asarray(total / objectives.shape[1], dtype=float)


def index_of_lowest_mean_objective(F: NDArray[Any]) -> int:
    """Row of the best solution of phase 1: the first one with the lowest mean objective, feasible or not.

    As jMetal's loop, a ``NaN`` mean is never the lowest, and a ``NaN`` mean in the first row keeps
    the first row.
    """
    means = mean_objectives(F)
    if means.shape[0] == 0:
        raise ValueError("ToP cannot choose the best solution of an empty population.")
    if np.isnan(means[0]):
        return 0
    return int(np.argmin(np.where(np.isnan(means), np.inf, means)))


def offspring_replaces_parent(
    offspring_violation: NDArray[Any],
    offspring_mean: NDArray[Any],
    parent_violation: NDArray[Any],
    parent_mean: NDArray[Any],
) -> NDArray[np.bool_]:
    """Whether each trial vector replaces the solution at its position: a smaller violation, or the same and a smaller mean objective."""
    offspring_cv = np.asarray(offspring_violation, dtype=float)
    parent_cv = np.asarray(parent_violation, dtype=float)
    smaller_mean = np.asarray(offspring_mean, dtype=float) < np.asarray(parent_mean, dtype=float)
    return np.asarray((offspring_cv < parent_cv) | ((offspring_cv == parent_cv) & smaller_mean), dtype=bool)


def feasible_ratio(violation: NDArray[Any]) -> float:
    """``Pf``: the number of rows with violation zero divided by the number of rows."""
    cv = np.asarray(violation, dtype=float)
    return int(np.count_nonzero(cv == 0.0)) / cv.shape[0]


def updated_delta(F: NDArray[Any], violation: NDArray[Any], delta: float) -> float:
    """``delta`` after a generation of phase 1, as jMetal's ``ToP.updatedDelta``.

    The objectives of the n feasible rows are normalized with their minimum and maximum, the
    normalized objectives of each row are added up, and ``delta`` is the ``floor(n / 3)``-th
    smallest sum minus the smallest one. With fewer than three feasible rows, ``delta`` keeps its
    value; an objective with a single value among them gives ``NaN``.
    """
    objectives = np.asarray(F, dtype=float)[np.asarray(violation, dtype=float) == 0.0]
    n = objectives.shape[0]
    if n < 3:
        return float(delta)
    minimum = np.min(objectives, axis=0)
    maximum = np.max(objectives, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = (objectives - minimum) / (maximum - minimum)
    sums = np.zeros(n, dtype=float)
    for j in range(normalized.shape[1]):
        sums = sums + normalized[:, j]
    sums = np.sort(sums)
    return float(sums[n // 3 - 1] - sums[0])


def phase_one_continues(
    delta: float,
    ratio: float,
    evaluations: int,
    max_evaluations: int,
    *,
    delta_threshold: float = DELTA_THRESHOLD,
    feasible_ratio_threshold: float = FEASIBLE_RATIO_THRESHOLD,
    phase_one_evaluation_fraction: float = PHASE_ONE_EVALUATION_FRACTION,
) -> bool:
    """Whether a generation that starts after ``evaluations`` belongs to phase 1.

    It does when ``(delta >= delta_threshold or ratio <= feasible_ratio_threshold)`` and
    ``evaluations <= phase_one_evaluation_fraction * max_evaluations``. A ``NaN`` delta fails its
    test.
    """
    searching = delta >= delta_threshold or ratio <= feasible_ratio_threshold
    return bool(searching and evaluations <= phase_one_evaluation_fraction * max_evaluations)


__all__ = [
    "CURRENT_TO_RAND_PROBABILITY",
    "DELTA_THRESHOLD",
    "FEASIBLE_RATIO_THRESHOLD",
    "INITIAL_DELTA",
    "PHASE_ONE_EVALUATION_FRACTION",
    "constraint_violation",
    "feasible_ratio",
    "index_of_lowest_mean_objective",
    "mean_objectives",
    "offspring_replaces_parent",
    "phase_one_continues",
    "updated_delta",
]
