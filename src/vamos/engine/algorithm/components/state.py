"""
Base state containers for algorithms.

Keeps shared fields and helpers that algorithm-specific states can extend.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

import numpy as np

if TYPE_CHECKING:
    from vamos.engine.algorithm.components.archive import (
        CrowdingDistanceArchive,
        HypervolumeArchive,
        MaxMinArchive,
        ReferenceDirectionsArchive,
        SPEA2Archive,
        UnboundedArchive,
    )
    from vamos.engine.algorithm.components.constraint_handling import ConstraintHandlingTechnique
    from vamos.engine.algorithm.components.termination import HVTracker
    from vamos.engine.hooks.genealogy import GenealogyTracker


_STALE = "_archive_rows_stale"


class ArchiveRows:
    """The ``archive_X`` and ``archive_F`` fields of a state: the external archive's rows.

    :func:`record_archive_rows` offers rows to a buffered archive without merging them and marks both
    fields stale; the first read of either merges the archive once (``contents()``) and keeps both.
    Assigning a value sets it, as for a plain field. As a dataclass default, it is ``None``.
    """

    def __set_name__(self, owner: type[Any], name: str) -> None:
        self._slot = f"_{name}_rows"

    @overload
    def __get__(self, instance: None, owner: type[Any]) -> None: ...

    @overload
    def __get__(self, instance: object, owner: type[Any] | None = None) -> np.ndarray[Any, Any] | None: ...

    def __get__(self, instance: object | None, owner: type[Any] | None = None) -> np.ndarray[Any, Any] | None:
        if instance is None:
            return None
        values = instance.__dict__
        if values.get(_STALE, False):
            _merge_archive_rows(instance)
        rows: np.ndarray[Any, Any] | None = values.get(self._slot)
        return rows

    def __set__(self, instance: object, value: np.ndarray[Any, Any] | None) -> None:
        if instance.__dict__.get(_STALE, False):
            _merge_archive_rows(instance)  # so that the other field keeps the archive's rows
        instance.__dict__[self._slot] = value


def _merge_archive_rows(state: object) -> None:
    values = state.__dict__
    values[_STALE] = False
    manager = values.get("archive_manager")
    if manager is not None:
        values["_archive_X_rows"], values["_archive_F_rows"] = manager.contents()


def record_archive_rows(
    state: Any,
    X: np.ndarray[Any, Any],
    F: np.ndarray[Any, Any],
    G: np.ndarray[Any, Any] | None,
) -> None:
    """Offer rows to the state's external archive, as ``archive_manager.update`` does.

    A buffered archive (``UnboundedArchive``) keeps them until ``archive_X`` or ``archive_F`` is read,
    or the result is built: a run that reads its archive only at the end filters it once instead of
    once per generation, with the same rows. Other archives, and states whose archive fields are
    plain attributes, are updated at once.
    """
    manager = state.archive_manager
    if manager is None:
        return
    if getattr(manager, "buffered", False) and isinstance(inspect.getattr_static(type(state), "archive_X", None), ArchiveRows):
        manager.add(X, F, G)
        state.__dict__[_STALE] = True
        return
    state.archive_X, state.archive_F = manager.update(X, F, G)


@dataclass
class AlgorithmState:
    """
    Base state container for evolutionary algorithms.

    This provides common fields that all algorithms need. Algorithm-specific
    subclasses can add additional fields as needed.
    """

    # Core population data
    X: np.ndarray[Any, Any]
    F: np.ndarray[Any, Any]
    G: np.ndarray[Any, Any] | None
    rng: np.random.Generator

    # Sizes
    pop_size: int = 100
    offspring_size: int = 100

    # Constraints. ``constraint_handling`` is None for cdp, which keeps each algorithm's own survival.
    constraint_mode: str = "none"
    constraint_handling: ConstraintHandlingTechnique | None = None

    # Generation tracking
    generation: int = 0
    n_eval: int = 0
    max_evals: int = 0

    # Archive (optional). The rows of a buffered archive are merged when first read (ArchiveRows).
    archive_size: int | None = None
    archive_X: ArchiveRows = ArchiveRows()
    archive_F: ArchiveRows = ArchiveRows()
    archive_manager: (
        CrowdingDistanceArchive | HypervolumeArchive | MaxMinArchive | ReferenceDirectionsArchive | SPEA2Archive | UnboundedArchive | None
    ) = None
    result_mode: str = "non_dominated"
    # What the live callback receives: "population", "nd_only" or "population_archive".
    live_callback_mode: str = "population"

    # Termination
    hv_tracker: HVTracker | None = None

    # Pending offspring for ask/tell
    pending_offspring: np.ndarray[Any, Any] | None = None
    pending_offspring_ids: np.ndarray[Any, Any] | None = None

    # Genealogy (optional)
    track_genealogy: bool = False
    genealogy_tracker: GenealogyTracker | None = None
    ids: np.ndarray[Any, Any] | None = None

    def hv_points(self) -> np.ndarray[Any, Any]:
        """Get points for hypervolume computation (archive if available, else population)."""
        if self.archive_F is not None and self.archive_F.size > 0:
            return self.archive_F
        return self.F


__all__ = ["AlgorithmState", "ArchiveRows", "record_archive_rows"]
