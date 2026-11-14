"""Constraint violation, shift-based density, fitness and stage of CMOEA-MS.

CMOEA-MS (Tian, Zhang, Su, Zhang, Tan and Jin, 2022) computes, at each generation, a fitness
of the union of the population and the offspring that is SPEA2's with a dominance relation that
depends on the stage. These functions transcribe jMetal's port (pull request jMetal/jMetal#526,
class ``CMOEAMSFitness`` and ``CMOEAMS.isStageB``), which follows the authors' implementation in
PlatEMO where it differs from the paper:

- the constraint violation of a solution is the mean, over the constraints, of its violation of
  each constraint divided by the largest violation of that constraint among the solutions; a
  constraint that no solution violates contributes zero;
- the shift-based density estimation (SDE) of a solution is ``1 / (d + 2)``, where ``d`` is the
  ``(floor(sqrt(n)) + 1)``-th smallest distance from the solution to the solutions shifted to it,
  itself included, on objectives normalized to [0, 1] among the solutions;
- the fitness is ``R + 1 / (d + 2)``: ``R`` is the sum of the strengths of the solutions that
  dominate a solution, and ``d`` the cosine distance to its ``floor(sqrt(n))``-th nearest other
  solution, so the solutions that no other one dominates have a fitness lower than 1;
- in stage A the dominance is Pareto dominance on the two objectives ``[SDE, CV]``; in stage B a
  smaller violation dominates, and equal violations are compared by Pareto dominance on the
  objectives;
- the algorithm is in stage B when the ratio of feasible solutions is greater than ``lambda`` and
  at least 10% of the evaluations have been done.

The divisions by zero of PlatEMO are kept, as jMetal keeps them: an objective with a single value
gives ``NaN`` densities, and a zero vector ``NaN`` cosine distances, which propagate as in MATLAB;
sorting puts ``NaN`` after every number, and a comparison with ``NaN`` is false. Sums run in the
order of jMetal's loops, so the values are jMetal's bit for bit. Solutions are rows.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IndexArray = NDArray[np.int_]

STAGE_B_EVALUATION_FRACTION = 0.1
"""Stage B needs at least this fraction of the evaluations done, as in PlatEMO's implementation."""


def normalized_violation(G: NDArray[Any] | None, n: int) -> FloatArray:
    """The constraint violation of each of the ``n`` rows, in [0, 1], as jMetal's ``CMOEAMSFitness.constraintViolation``.

    ``G`` holds VAMOS's constraint values, violated where positive (``None`` without constraints,
    which gives zeros). The violation of each constraint, ``max(g, 0)`` with ``NaN`` counted as no
    violation, is divided by its largest value among the rows; a constraint whose largest
    violation is zero (``0 / 0`` in PlatEMO, detected by the ``NaN`` of the first row) contributes
    zero. The violation of a row is the mean over the constraints.
    """
    if G is None:
        return np.zeros(int(n), dtype=float)
    values = np.asarray(G, dtype=float).reshape(int(n), -1)
    n_constraints = values.shape[1]
    if n_constraints == 0 or values.shape[0] == 0:
        return np.zeros(values.shape[0], dtype=float)
    # max(g, 0) of MATLAB ignores a NaN, so a NaN value counts as no violation.
    violation = np.where(values > 0.0, values, 0.0)
    total = np.zeros(values.shape[0], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        for j in range(n_constraints):
            column = violation[:, j] / violation[:, j].max()
            if math.isnan(column[0]):
                column = np.zeros_like(column)
            total = total + column
    return np.asarray(total / n_constraints, dtype=float)


def _normalized_objectives(F: NDArray[Any]) -> FloatArray:
    """The objectives scaled to [0, 1] with their minimum and maximum, which ignore ``NaN`` as MATLAB's ``min`` and ``max``."""
    objectives = np.asarray(F, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        minimum = np.fmin.reduce(objectives, axis=0)
        maximum = np.fmax.reduce(objectives, axis=0)
        return np.asarray((objectives - minimum) / (maximum - minimum), dtype=float)


def shift_based_density(F: NDArray[Any]) -> FloatArray:
    """The SDE of each row, ``1 / (d + 2)``, as jMetal's ``CMOEAMSFitness.shiftBasedDensity``.

    The objectives are normalized to [0, 1] among the rows. Each other row is shifted to row
    ``i``: its objectives that are better than those of ``i`` take the values of ``i``. ``d`` is the
    ``(floor(sqrt(n)) + 1)``-th smallest Euclidean distance between ``i`` and the shifted rows,
    counting the distance of ``i`` to itself, so a lower value is a less crowded row. An objective
    with a single value gives ``NaN`` to every row, as in PlatEMO. Needs at least two rows.
    """
    normalized = _normalized_objectives(F)
    n, n_obj = normalized.shape
    if n < 2:
        raise ValueError("CMOEA-MS's shift-based density needs at least two solutions.")
    squares = np.zeros((n, n), dtype=float)
    for j in range(n_obj):
        own = normalized[:, j][:, None]
        other = normalized[:, j][None, :]
        # Row i, column k: solution k shifted to solution i.
        shifted = np.where(other < own, own, other)
        difference = own - shifted
        squares = squares + difference * difference
    distances = np.sort(np.sqrt(squares), axis=1)
    neighbour = int(math.floor(math.sqrt(n)))
    return np.asarray(1.0 / (distances[:, neighbour] + 2.0), dtype=float)


def cosine_distances(points: NDArray[Any]) -> FloatArray:
    """One minus the cosine of the angle between each pair of rows, as MATLAB's ``pdist2(X, X, 'cosine')`` and jMetal.

    Each row is divided by its norm, computed after scaling it by its largest absolute value
    (which ignores ``NaN``); the cosine is limited to 1 against round-off errors. A zero row, or one
    with a ``NaN``, has ``NaN`` distances, its distance to itself included.
    """
    values = np.asarray(points, dtype=float)
    n, n_dim = values.shape
    magnitude = np.abs(values)
    largest = np.where(np.isnan(magnitude), 0.0, magnitude).max(axis=1) if n_dim else np.zeros(n)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        scaled = values / largest[:, None]
        sum_of_squares: FloatArray = np.zeros(n, dtype=float)
        for k in range(n_dim):
            sum_of_squares = np.asarray(sum_of_squares + scaled[:, k] * scaled[:, k], dtype=float)
        norm = np.where(largest > 0.0, np.sqrt(sum_of_squares) * largest, 0.0)
        unit = values / norm[:, None]
        cosine: FloatArray = np.zeros((n, n), dtype=float)
        for k in range(n_dim):
            cosine = np.asarray(cosine + np.multiply.outer(unit[:, k], unit[:, k]), dtype=float)
    cosine = np.where(cosine > 1.0, 1.0, cosine)
    return np.asarray(1.0 - cosine, dtype=float)


def dominance_matrix(points: NDArray[Any], violation: NDArray[Any]) -> NDArray[np.bool_]:
    """``dom[i, j]`` is true when row ``i`` dominates row ``j`` in jMetal's ``CMOEAMSFitness.fitness``.

    ``i`` dominates ``j`` if its violation is smaller or, when neither violation is smaller (equal,
    or one of them ``NaN``), if it is better in some coordinate of ``points`` and worse in none (a
    comparison with ``NaN`` is false), as ``any(a < b) - any(a > b)`` of PlatEMO.
    """
    values = np.asarray(points, dtype=float)
    cv = np.asarray(violation, dtype=float)
    smaller = cv[:, None] < cv[None, :]
    larger = cv[:, None] > cv[None, :]
    better = np.any(values[:, None, :] < values[None, :, :], axis=2)
    worse = np.any(values[:, None, :] > values[None, :, :], axis=2)
    return np.asarray(smaller | (~smaller & ~larger & better & ~worse), dtype=bool)


def spea2_fitness(points: NDArray[Any], violation: NDArray[Any]) -> FloatArray:
    """The fitness ``R + D`` of PlatEMO's ``CalFitness`` for each row, as jMetal's ``CMOEAMSFitness.fitness``.

    ``R`` is the sum of the strengths (the number of rows a row dominates, see
    :func:`dominance_matrix`) of the rows that dominate a row, zero for the non-dominated ones.
    ``D = 1 / (d + 2)``, where ``d`` is the :func:`cosine distance <cosine_distances>` between the
    vectors of ``points`` to the ``floor(sqrt(n))``-th nearest other row, is in (0, 0.5] (``NaN``
    sorts after every distance), so the non-dominated rows have a fitness lower than 1. A lone row
    has ``D = 0``.
    """
    values = np.asarray(points, dtype=float)
    n = values.shape[0]
    if n == 0:
        raise ValueError("CMOEA-MS's fitness needs at least one solution.")
    dominates = dominance_matrix(values, violation)
    strength = dominates.sum(axis=1, dtype=np.int64)
    raw = (dominates.T.astype(np.int64) @ strength).astype(float)
    distances = cosine_distances(values)
    np.fill_diagonal(distances, np.inf)
    neighbour = int(math.floor(math.sqrt(n))) - 1
    kth = np.sort(distances, axis=1)[:, neighbour]
    return np.asarray(raw + 1.0 / (kth + 2.0), dtype=float)


def stage_a_fitness(F: NDArray[Any], violation: NDArray[Any]) -> FloatArray:
    """The fitness of stage A: that of the points ``[SDE, CV]`` with Pareto dominance (PlatEMO's ``CalFitness([CalSDE(objs)', CV])``)."""
    cv = np.asarray(violation, dtype=float)
    points = np.column_stack([shift_based_density(F), cv])
    return spea2_fitness(points, np.zeros(cv.shape[0], dtype=float))


def stage_b_fitness(F: NDArray[Any], violation: NDArray[Any]) -> FloatArray:
    """The fitness of stage B: that of the objectives with the smaller violation first (PlatEMO's ``CalFitness(objs, CV)``)."""
    return spea2_fitness(F, violation)


def is_stage_b(violation: NDArray[Any], threshold: float, evaluations: int, max_evaluations: int) -> bool:
    """Whether the union is in stage B, as jMetal's ``CMOEAMS.isStageB``.

    The ratio of feasible rows (a violation of zero; ``NaN`` is not feasible) must be greater than
    ``threshold`` (``lambda``), and ``evaluations``, the evaluations done with those of the
    offspring, at least 10% of ``max_evaluations``.
    """
    cv = np.asarray(violation, dtype=float)
    ratio = int(np.count_nonzero(cv <= 0.0)) / cv.shape[0]
    return bool(ratio > threshold and evaluations >= STAGE_B_EVALUATION_FRACTION * max_evaluations)


def sorted_indices(values: NDArray[Any]) -> IndexArray:
    """The indices that sort ``values`` in ascending order, as ``[~, I] = sort(values)`` of MATLAB: stable, ``NaN`` last."""
    return np.asarray(np.argsort(np.asarray(values, dtype=float), kind="stable"), dtype=int)


__all__ = [
    "STAGE_B_EVALUATION_FRACTION",
    "cosine_distances",
    "dominance_matrix",
    "is_stage_b",
    "normalized_violation",
    "shift_based_density",
    "sorted_indices",
    "spea2_fitness",
    "stage_a_fitness",
    "stage_b_fitness",
]
