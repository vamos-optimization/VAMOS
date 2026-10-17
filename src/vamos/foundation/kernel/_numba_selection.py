from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .numba_ops import njit


@njit(cache=True)
def binary_tournament_winners_numba(
    ranks: NDArray[np.number[Any]],
    crowding: NDArray[np.float64],
    first: NDArray[np.integer[Any]],
    second: NDArray[np.integer[Any]],
) -> NDArray[np.int64]:
    """Resolve binary tournaments by rank, then crowding; mark full ties with -1.

    The caller resolves the ties from its own generator so that random draws
    match the NumPy reference: one coin per tied tournament, in order.
    """
    n_parents = first.shape[0]
    winners = np.empty(n_parents, dtype=np.int64)

    for i in range(n_parents):
        left = first[i]
        right = second[i]
        left_rank = ranks[left]
        right_rank = ranks[right]

        if left_rank < right_rank:
            winners[i] = left
            continue
        if right_rank < left_rank:
            winners[i] = right
            continue

        left_crowding = crowding[left]
        right_crowding = crowding[right]

        if np.isnan(left_crowding):
            left_crowding = -np.inf
        if np.isnan(right_crowding):
            right_crowding = -np.inf

        if left_crowding > right_crowding:
            winners[i] = left
            continue
        if right_crowding > left_crowding:
            winners[i] = right
            continue

        winners[i] = -1

    return winners


__all__ = ["binary_tournament_winners_numba"]
