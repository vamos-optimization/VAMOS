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
      row by row (SPEA2's environmental archive, SMPSO's swarm). NSGA-II sends its first front
      instead with its default ``live_callback_mode="nd_only"``, and its population with
      ``"population"``;
    - ``stats["evals"]`` is the number of evaluations used so far.

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
