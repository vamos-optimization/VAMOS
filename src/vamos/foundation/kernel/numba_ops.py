from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast

import numpy as np
from numba import njit as _numba_njit
from numpy.typing import NDArray

_F = TypeVar("_F", bound=Callable[..., object])


def njit(*args: Any, **kwargs: Any) -> Callable[[_F], _F]:
    """Typed wrapper around numba.njit to keep mypy happy."""
    return cast(Callable[[_F], _F], _numba_njit(*args, **kwargs))


@njit(cache=True)
def tournament_winners_numba(
    ranks: NDArray[np.int64] | NDArray[np.float64],
    crowding: NDArray[np.float64],
    candidates: NDArray[np.int64],
    tie_break: NDArray[np.float64],
) -> NDArray[np.int64]:
    """Resolve tournament winners for arbitrary pressure using caller-provided candidates."""
    n_parents, pressure = candidates.shape
    winners = np.empty(n_parents, dtype=np.int64)

    for i in range(n_parents):
        best = candidates[i, 0]
        best_rank = ranks[best]
        best_crowding = crowding[best]
        if np.isnan(best_crowding):
            best_crowding = -np.inf
        best_key = tie_break[i, 0]

        for j in range(1, pressure):
            cand = candidates[i, j]
            cand_rank = ranks[cand]
            cand_crowding = crowding[cand]
            if np.isnan(cand_crowding):
                cand_crowding = -np.inf
            cand_key = tie_break[i, j]

            if cand_rank < best_rank:
                best = cand
                best_rank = cand_rank
                best_crowding = cand_crowding
                best_key = cand_key
                continue
            if cand_rank > best_rank:
                continue
            if cand_crowding > best_crowding:
                best = cand
                best_crowding = cand_crowding
                best_key = cand_key
                continue
            if cand_crowding < best_crowding:
                continue
            if cand_key < best_key:
                best = cand
                best_key = cand_key

        winners[i] = best

    return winners
