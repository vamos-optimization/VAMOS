"""Keep the feasible solutions of loaded fronts, or those inside objective ranges, before plotting and ranking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from vamos.ux.studio.data import FrontRecord, normalize_objectives
from vamos.ux.studio.dm import DecisionView, feasible_indices, filter_by_objective_ranges

ObjectiveRange = tuple[float | None, float | None]


def objective_bounds(fronts: Sequence[FrontRecord]) -> list[tuple[float, float]]:
    """The smallest and largest value of each objective shared by ``fronts``."""
    arrays = [front.points_F for front in fronts if front.points_F.size]
    if not arrays:
        return []
    n_obj = min(array.shape[1] for array in arrays)
    stacked = np.vstack([array[:, :n_obj] for array in arrays])
    return [(float(stacked[:, index].min()), float(stacked[:, index].max())) for index in range(n_obj)]


def have_constraints(fronts: Sequence[FrontRecord]) -> bool:
    return any(front.constraints is not None for front in fronts)


def filter_front(front: FrontRecord, *, feasible_only: bool = False, ranges: Sequence[ObjectiveRange] = ()) -> FrontRecord:
    """The solutions of ``front`` that are feasible, when asked, and lie in every objective range."""
    n_points, n_obj = front.points_F.shape
    if not n_points:
        return front
    view = DecisionView(front=front, normalized_F=normalize_objectives(front.points_F), decoded_X=None, constraints=front.constraints)
    keep: NDArray[np.intp] = np.arange(n_points)
    if feasible_only:
        keep = np.intersect1d(keep, feasible_indices(view))
    if any(bound is not None for pair in ranges for bound in pair):
        padded = [*list(ranges)[:n_obj], *[(None, None)] * max(n_obj - len(ranges), 0)]
        keep = np.intersect1d(keep, filter_by_objective_ranges(view, padded))
    if keep.size == n_points:
        return front
    return replace(
        front,
        points_F=front.points_F[keep],
        points_X=None if front.points_X is None else front.points_X[keep],
        constraints=None if front.constraints is None else front.constraints[keep],
    )


def filter_fronts(
    fronts: Sequence[FrontRecord], *, feasible_only: bool = False, ranges: Sequence[ObjectiveRange] = ()
) -> list[FrontRecord]:
    """Filter every front, leaving out the fronts with no solution left."""
    filtered = [filter_front(front, feasible_only=feasible_only, ranges=ranges) for front in fronts]
    return [front for front in filtered if front.points_F.shape[0]]


def solution_count(fronts: Sequence[FrontRecord]) -> int:
    return sum(int(front.points_F.shape[0]) for front in fronts)


__all__ = ["ObjectiveRange", "filter_front", "filter_fronts", "have_constraints", "objective_bounds", "solution_count"]
