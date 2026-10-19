"""Follow one optimization run from another thread, and stop it cooperatively.

``LiveRun`` is a live-visualization callback: the algorithms call it after
every generation, so a Studio page can read the latest front and evaluation
count while the run executes in a background thread. Every built-in
algorithm asks ``should_stop()`` after a generation, so ``request_stop()``
ends the run at the next generation boundary with its current population.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

MAX_SNAPSHOTS = 1000


class LiveRun:
    """The fronts, generation and evaluation count of a running optimization."""

    def __init__(self, budget: int) -> None:
        self.budget = int(budget)
        self.generation = 0
        self.evaluations = 0
        self.started = time.monotonic()
        self.finished_at: float | None = None
        self.snapshots: list[tuple[int, NDArray[np.float64]]] = []
        self._stride = 1
        self._stop = threading.Event()

    # ---- live-visualization protocol ----

    def on_start(self, ctx: Any = None) -> None:
        self.started = time.monotonic()

    def on_generation(self, generation: int, F: Any = None, X: Any = None, stats: dict[str, Any] | None = None) -> None:
        self.generation = int(generation)
        evaluations = (stats or {}).get("evals")
        if isinstance(evaluations, (int, np.integer)):
            self.evaluations = int(evaluations)
        if F is not None and generation % self._stride == 0:
            self.snapshots.append((int(generation), np.array(F, dtype=float)))
            if len(self.snapshots) > MAX_SNAPSHOTS:
                # Keep memory bounded on long runs: drop every other snapshot and halve the rate.
                self.snapshots = self.snapshots[::2]
                self._stride *= 2

    def on_end(self, final_F: Any = None, final_stats: dict[str, Any] | None = None) -> None:
        evaluations = (final_stats or {}).get("evals")
        if isinstance(evaluations, (int, np.integer)):
            self.evaluations = int(evaluations)
        if final_F is not None:
            self.snapshots.append((self.generation, np.array(final_F, dtype=float)))
        self.finished_at = time.monotonic()

    def should_stop(self) -> bool:
        return self._stop.is_set()

    # ---- control and reading ----

    def request_stop(self) -> None:
        self._stop.set()

    @property
    def stop_requested(self) -> bool:
        return self._stop.is_set()

    @property
    def elapsed(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return max(0.0, end - self.started)

    @property
    def fraction(self) -> float:
        return min(1.0, self.evaluations / self.budget) if self.budget > 0 else 0.0

    def latest(self) -> NDArray[np.float64] | None:
        return self.snapshots[-1][1] if self.snapshots else None


__all__ = ["MAX_SNAPSHOTS", "LiveRun"]
