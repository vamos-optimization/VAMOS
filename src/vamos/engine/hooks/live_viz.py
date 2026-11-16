from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.observer import RunContext


class LiveVisualization(Protocol):
    """Callback interface for live/streaming visualization.

    Every built-in algorithm calls ``on_generation`` with one payload:

    - ``generation`` is 0 right after the initial population is evaluated and grows by one
      per generation; steady-state algorithms (SMS-EMOA, MOEA/D) report one generation per
      ``pop_size`` evaluations;
    - ``F`` and ``X`` are the objective and decision vectors of the whole current population,
      row by row (SPEA2's environmental archive, SMPSO's swarm), with the configuration's
      ``live_callback_mode="population"``, the default of every algorithm but NSGA-II;
      ``"nd_only"``, NSGA-II's default, sends the population's first front instead;
    - ``stats["evals"]`` is the number of evaluations used so far;
    - with ``live_callback_mode="population_archive"``, ``stats["archive"]`` also holds the
      external archive, ``{"X": ..., "F": ...}``, which is what a run with an archive returns
      (SMPSO's leaders when it has no external archive); without an archive the key is absent.

    An exception raised by a callback propagates and stops the run.
    """

    def on_start(self, ctx: RunContext | None = None) -> None: ...

    def on_generation(
        self,
        generation: int,
        F: NDArray[np.float64] | None = None,
        X: NDArray[np.generic] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None: ...

    def on_end(
        self,
        final_F: NDArray[np.float64] | None = None,
        final_stats: dict[str, Any] | None = None,
    ) -> None: ...


class NoOpLiveVisualization:
    """Default no-op implementation."""

    def on_start(self, ctx: RunContext | None = None) -> None:
        return None

    def on_generation(
        self,
        generation: int,
        F: NDArray[np.float64] | None = None,
        X: NDArray[np.generic] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        return None

    def on_end(
        self,
        final_F: NDArray[np.float64] | None = None,
        final_stats: dict[str, Any] | None = None,
    ) -> None:
        return None


__all__ = ["LiveVisualization", "NoOpLiveVisualization"]
