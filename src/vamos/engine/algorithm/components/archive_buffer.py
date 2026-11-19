"""The row buffer of the unbounded archive: rows offered with ``add`` are merged when they are needed.

``_BaseArchive`` (``archive_core``) inherits it. A buffered archive whose contents are clean keeps the
rows it is offered and merges them in one update when its contents are read, when ``update`` is
called, or when the kept rows outnumber ``max(PENDING_ROWS, 2 * size)``, with the contents of one
update per call.
"""

from __future__ import annotations

from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.constraints.utils import is_feasible

FloatArray: TypeAlias = NDArray[np.float64]
DecisionArray: TypeAlias = NDArray[np.generic]

# A buffered archive merges the rows offered with ``add`` once they outnumber this, or twice its size.
PENDING_ROWS = 4096


class RowBuffer:
    """The rows that ``add`` keeps for a later merge, and their checks; mixed into ``_BaseArchive``."""

    buffered: bool
    capacity: int | None
    covers_last_input: bool
    unchanged_by_last_input: bool
    last_capacity_pruned: int
    _pending: list[tuple[DecisionArray, FloatArray, FloatArray | None]]
    _pending_rows: int
    _clean: bool
    _size: int
    _G: FloatArray | None
    _dtype: np.dtype[Any]
    _n_var: int
    _n_obj: int

    def update(
        self,
        population_X: DecisionArray,
        population_F: NDArray[np.generic],
        population_G: NDArray[np.generic] | None = None,
    ) -> tuple[DecisionArray, FloatArray]:
        raise NotImplementedError

    def _update_now(
        self,
        population_X: DecisionArray,
        population_F: NDArray[np.generic],
        population_G: NDArray[np.generic] | None = None,
    ) -> tuple[DecisionArray, FloatArray]:
        raise NotImplementedError

    def add(
        self,
        population_X: DecisionArray,
        population_F: NDArray[np.generic],
        population_G: NDArray[np.generic] | None = None,
    ) -> None:
        """Offer rows as ``update`` does, without computing the contents.

        A ``buffered`` archive with clean contents (mutually non-dominated, finite, feasible when it holds
        constraint values, deduplicated with a zero objective tolerance) keeps the rows whose feasible
        objective vectors are finite, and merges them when ``contents``, ``constraint_values`` or
        ``update`` needs them, or when they outnumber ``max(PENDING_ROWS, 2 * size)``. One merge of many
        rows gives the contents, in the order, of one ``update`` per call: a clean archive keeps the
        feasible non-dominated rows in the order they were offered, and the first of equal ones. Other
        rows, or rows offered to an archive that is not buffered or not clean, are merged at once.
        """
        pop_X, pop_F, pop_G = self._checked_rows(population_X, population_F, population_G)
        if self.buffered and pop_X.shape[0] > 0 and self._bufferable(pop_F, pop_G):
            # Copies: the caller may write into its population after offering it.
            self._pending.append((pop_X.copy(), pop_F.copy(), None if pop_G is None else pop_G.copy()))
            self._pending_rows += pop_X.shape[0]
            # As after an update of clean contents: every offered row will be archived or covered.
            self.last_capacity_pruned = 0
            self.covers_last_input = True
            self.unchanged_by_last_input = False
            if self._pending_rows > max(PENDING_ROWS, 2 * self._size):
                self._flush()
            return
        self.update(pop_X, pop_F, pop_G)

    def _bufferable(self, pop_F: FloatArray, pop_G: FloatArray | None) -> bool:
        """Whether ``add`` may keep these rows for a later merge (see ``add``)."""
        if not self._clean or self._size == 0 or self.capacity is not None or (self._G is None) != (pop_G is None):
            return False
        if pop_G is None:
            return bool(np.isfinite(pop_F).all())
        assert self._G is not None
        if pop_G.shape[1] != self._G.shape[1]:
            return False
        return bool(np.isfinite(pop_F[is_feasible(pop_G, n=pop_G.shape[0])]).all())

    def _flush(self) -> None:
        """Merge the rows that ``add`` kept, as one update."""
        if not self._pending:
            return
        pending = self._pending
        self._pending = []
        self._pending_rows = 0
        X = np.concatenate([rows[0] for rows in pending])
        F = np.concatenate([rows[1] for rows in pending])
        Gs = [rows[2] for rows in pending if rows[2] is not None]
        G = np.concatenate(Gs) if Gs else None
        self._update_now(X, F, G)
        # As after the calls that offered them: the merge covered every row.
        self.covers_last_input = self._clean
        self.unchanged_by_last_input = False

    def _checked_rows(
        self,
        population_X: DecisionArray,
        population_F: NDArray[np.generic],
        population_G: NDArray[np.generic] | None,
    ) -> tuple[DecisionArray, FloatArray, FloatArray | None]:
        pop_X = np.asarray(population_X, dtype=self._dtype, order="C")
        pop_F = np.asarray(population_F, dtype=float, order="C")
        if pop_X.ndim != 2:
            raise ValueError("population_X must be a 2D array.")
        if pop_F.ndim != 2:
            raise ValueError("population_F must be a 2D array.")
        if pop_X.shape[0] != pop_F.shape[0]:
            raise ValueError("population_X and population_F must have the same number of rows.")
        if pop_X.shape[1] != self._n_var:
            raise ValueError(f"population_X has {pop_X.shape[1]} columns, expected {self._n_var}.")
        if pop_F.shape[1] != self._n_obj:
            raise ValueError(f"population_F has {pop_F.shape[1]} columns, expected {self._n_obj}.")
        pop_G: FloatArray | None = None
        if population_G is not None and pop_X.shape[0] > 0:
            pop_G = np.asarray(population_G, dtype=float, order="C")
            if pop_G.ndim != 2:
                raise ValueError("population_G must be a 2D array when provided.")
            if pop_G.shape[0] != pop_X.shape[0]:
                raise ValueError("population_G must have the same number of rows as population_X.")
        return pop_X, pop_F, pop_G


__all__ = ["PENDING_ROWS", "RowBuffer"]
