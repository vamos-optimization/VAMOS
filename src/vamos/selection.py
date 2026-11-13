"""Deterministic environmental selection over evaluated objective vectors."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from vamos.foundation.kernel.numpy_backend import NumPyKernel


@dataclass(frozen=True, slots=True)
class EnvironmentalSelectionResult:
    """Immutable NSGA-II environmental-selection result.

    Objectives supplied to :func:`select_survivors` follow the minimization
    convention. Candidate objects and identifiers remain the caller's
    responsibility; this result preserves their original integer positions.

    Experimental: exported by ``vamos`` and ``vamos.api``, but not covered by
    the 1.x stability policy.
    """

    selected_indices: tuple[int, ...]
    ranks: tuple[int, ...]
    crowding_distances: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate the public result contract."""
        if type(self.selected_indices) is not tuple:
            raise TypeError("selected_indices must be a built-in tuple")
        if type(self.ranks) is not tuple:
            raise TypeError("ranks must be a built-in tuple")
        if type(self.crowding_distances) is not tuple:
            raise TypeError("crowding_distances must be a built-in tuple")
        if not self.ranks:
            raise ValueError("ranks must contain at least one candidate")
        if len(self.crowding_distances) != len(self.ranks):
            raise ValueError("ranks and crowding_distances must have equal length")
        if not self.selected_indices:
            raise ValueError("selected_indices must contain at least one candidate")
        if len(self.selected_indices) > len(self.ranks):
            raise ValueError("selected_indices cannot exceed the candidate count")

        if any(type(index) is not int for index in self.selected_indices):
            raise TypeError("selected_indices must contain built-in integers")
        if len(set(self.selected_indices)) != len(self.selected_indices):
            raise ValueError("selected_indices must be unique")
        if any(index < 0 or index >= len(self.ranks) for index in self.selected_indices):
            raise ValueError("selected_indices must refer to valid candidate positions")

        if any(type(rank) is not int for rank in self.ranks):
            raise TypeError("ranks must contain built-in integers")
        if any(rank < 0 for rank in self.ranks):
            raise ValueError("ranks must be nonnegative")

        if any(type(distance) is not float for distance in self.crowding_distances):
            raise TypeError("crowding_distances must contain built-in floats")
        if any(math.isnan(distance) or distance < 0.0 for distance in self.crowding_distances):
            raise ValueError("crowding_distances must be nonnegative finite values or positive infinity")


def _objective_matrix(objectives: ArrayLike) -> NDArray[np.float64]:
    """Return a validated, independent minimization-objective matrix."""
    try:
        matrix = np.array(objectives, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("objectives must be a rectangular numeric matrix") from exc

    if matrix.ndim != 2:
        raise ValueError("objectives must be a two-dimensional matrix")
    if matrix.shape[0] == 0:
        raise ValueError("objectives must contain at least one candidate")
    if matrix.shape[1] < 2:
        raise ValueError("objectives must contain at least two objectives")
    if not np.isfinite(matrix).all():
        raise ValueError("objectives must contain only finite values")
    return matrix


def select_survivors(
    objectives: ArrayLike,
    survivor_count: int,
) -> EnvironmentalSelectionResult:
    """Select survivors from already evaluated minimization objectives.

    The function performs NSGA-II nondominated ranking and crowding-distance
    calculation only. It performs no initialization, evaluation, variation,
    repair, optimization loop, or random operation.

    Candidates are ordered by lower rank, higher crowding distance, then lower
    original input index. The returned indices therefore map directly back to
    the caller's original candidate sequence.

    Experimental: exported by ``vamos`` and ``vamos.api``, but not covered by
    the 1.x stability policy.
    """
    matrix = _objective_matrix(objectives)
    candidate_count = matrix.shape[0]

    if type(survivor_count) is not int:
        raise TypeError("survivor_count must be a built-in integer")
    if survivor_count < 1 or survivor_count > candidate_count:
        raise ValueError("survivor_count must be between one and the number of candidates")

    ranks, crowding = NumPyKernel().nsga2_ranking(matrix)
    original_indices = np.arange(candidate_count, dtype=int)
    ordered = np.lexsort((original_indices, -crowding, ranks))
    selected = ordered[:survivor_count]

    result = EnvironmentalSelectionResult(
        selected_indices=tuple(int(index) for index in selected),
        ranks=tuple(int(rank) for rank in ranks),
        crowding_distances=tuple(float(distance) for distance in crowding),
    )
    if len(result.selected_indices) != survivor_count:
        raise RuntimeError("environmental selection returned an invalid survivor count")
    return result


__all__ = ["EnvironmentalSelectionResult", "select_survivors"]
