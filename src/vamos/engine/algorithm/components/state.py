"""
Base state containers for algorithms.

Keeps shared fields and helpers that algorithm-specific states can extend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

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

    # Archive (optional)
    archive_size: int | None = None
    archive_X: np.ndarray[Any, Any] | None = None
    archive_F: np.ndarray[Any, Any] | None = None
    archive_manager: (
        CrowdingDistanceArchive | HypervolumeArchive | MaxMinArchive | ReferenceDirectionsArchive | SPEA2Archive | UnboundedArchive | None
    ) = None
    result_mode: str = "non_dominated"

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


__all__ = ["AlgorithmState"]
