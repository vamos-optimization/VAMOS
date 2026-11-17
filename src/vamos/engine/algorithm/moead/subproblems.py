"""The order in which MOEA/D visits its subproblems, and the parents it draws for each.

**VAMOS 1.4.0:** ``subproblem_order`` chooses the order, as Evolver's
``subProblemIdGenerator`` does:

- ``"random_permutation"`` (the default, VAMOS's only order before): a new random permutation of
  the subproblems in each cycle, drawn when the previous one is used up, as jMetal's
  ``RandomPermutationCycle``;
- ``"cyclic"``: 0, 1, ..., ``pop_size - 1``, then 0 again, as jMetal's ``CyclicIntegerSequence``.
  It draws no random number.

The DE schemes rand/2 need four random parents, drawn as the first two are: from the
neighbourhood or from the population, all distinct.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from vamos.foundation.exceptions import ConfigurationError

if TYPE_CHECKING:
    from .state import MOEADState

SUBPROBLEM_ORDERS: tuple[str, ...] = ("random_permutation", "cyclic")


def normalize_subproblem_order(value: object) -> str:
    """``value`` in lower case, or ``ConfigurationError`` unless it is one of :data:`SUBPROBLEM_ORDERS`."""
    name = value.strip().lower() if isinstance(value, str) else None
    if name not in SUBPROBLEM_ORDERS:
        options = ", ".join(repr(order) for order in SUBPROBLEM_ORDERS)
        raise ConfigurationError(f"subproblem_order must be one of {options}; got {value!r}.")
    return name


def new_subproblem_order(rng: np.random.Generator, pop_size: int, cyclic: bool) -> np.ndarray[Any, Any]:
    """The order of the next cycle: ``0 .. pop_size - 1``, or a permutation drawn from ``rng``."""
    if cyclic:
        return np.arange(pop_size, dtype=int)
    return rng.permutation(pop_size).astype(int, copy=False)


def checkpoint_subproblem_order(order_raw: object, pop_size: int, cyclic: bool, rng: np.random.Generator) -> np.ndarray[Any, Any]:
    """The subproblem order of a checkpoint, or a new one when it has none."""
    if order_raw is None:
        return new_subproblem_order(rng, pop_size, cyclic)
    order = np.asarray(order_raw, dtype=int)
    if order.ndim != 1 or not np.array_equal(np.sort(order), np.arange(pop_size)):
        raise ValueError(f"Checkpoint subproblem_order is not a permutation of {pop_size} subproblems.")
    if cyclic and not np.array_equal(order, np.arange(pop_size)):
        raise ValueError("Checkpoint subproblem_order is a random permutation; the run asks for subproblem_order='cyclic'.")
    return order


def next_active_indices(st: MOEADState, batch_size: int) -> np.ndarray[Any, Any]:
    """The next ``batch_size`` subproblems of the order, starting a new cycle when one is used up."""
    if st.subproblem_order.size != st.pop_size:
        st.subproblem_order = new_subproblem_order(st.rng, st.pop_size, st.subproblem_cyclic)
        st.subproblem_cursor = 0

    active = np.empty(batch_size, dtype=int)
    for i in range(batch_size):
        if st.subproblem_cursor >= st.pop_size:
            st.subproblem_order = new_subproblem_order(st.rng, st.pop_size, st.subproblem_cyclic)
            st.subproblem_cursor = 0
        active[i] = int(st.subproblem_order[st.subproblem_cursor])
        st.subproblem_cursor += 1
    return active


def _distinct(draws: list[np.ndarray[Any, Any]]) -> list[np.ndarray[Any, Any]]:
    """Distinct positions in a pool from draws ``d_j`` in ``[0, size - j)``, one array per parent.

    Position ``j`` is the ``d_j``-th of the positions not taken yet: the draw skips, in increasing
    order, each earlier position it reaches. This is uniform sampling without replacement; with
    two draws the second skips the first, as MOEA/D has always drawn its pair.
    """
    chosen: list[np.ndarray[Any, Any]] = []
    for draw in draws:
        index = draw.copy()
        if chosen:
            for earlier in np.sort(np.stack(chosen), axis=0):
                index += index >= earlier
        chosen.append(index)
    return chosen


def sample_parents(
    st: MOEADState,
    active: np.ndarray[Any, Any],
    use_neighbors: np.ndarray[Any, Any],
    out: np.ndarray[Any, Any],
    rows: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    """Draw ``out.shape[1]`` distinct parents for each active subproblem, from its neighbourhood or the population.

    The draws of each parent come one after another for all the children: the first parents of
    the neighbourhood children, the second ones, ..., then those of the population children.
    ``rows`` is ``arange`` of at least the number of neighbourhood children.
    """
    count = out.shape[1]
    neighbor_mask = np.asarray(use_neighbors, dtype=bool)
    if bool(np.any(neighbor_mask)):
        active_neighbors = st.neighbors[active[neighbor_mask]]
        local_count, local_size = active_neighbors.shape
        draws = [st.rng.integers(0, local_size - j, size=local_count, dtype=np.int64) for j in range(count)]
        for slot, local in enumerate(_distinct(draws)):
            out[neighbor_mask, slot] = active_neighbors[rows[:local_count], local]
    global_mask = ~neighbor_mask
    if bool(np.any(global_mask)):
        global_count = int(np.sum(global_mask))
        draws = [st.rng.integers(0, st.pop_size - j, size=global_count, dtype=np.int64) for j in range(count)]
        for slot, index in enumerate(_distinct(draws)):
            out[global_mask, slot] = index
    return out


__all__ = [
    "SUBPROBLEM_ORDERS",
    "checkpoint_subproblem_order",
    "new_subproblem_order",
    "next_active_indices",
    "normalize_subproblem_order",
    "sample_parents",
]
