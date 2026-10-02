"""Fold progress events into the state rendered by the run view.

This module is UI-independent: the interface polls a :class:`ProgressMonitor`
and redraws only when :meth:`ProgressMonitor.poll` reports new events.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .events import TERMINAL_STATUSES, read_events


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


@dataclass
class ProgressState:
    """Latest known state of one GUI run."""

    status: str = "pending"
    request: dict[str, Any] = field(default_factory=dict)
    elapsed: float = 0.0
    generation: int | None = None
    updates: int = 0
    evaluations: int | None = None
    n_obj: int | None = None
    front: npt.NDArray[np.float64] | None = None
    front_size: int = 0
    convergence_x: list[float] = field(default_factory=list)
    convergence_y: list[float] = field(default_factory=list)
    run_dir: str | None = None
    error: str | None = None
    malformed_lines: int = 0

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def max_evaluations(self) -> int | None:
        return _optional_int(self.request.get("max_evaluations"))

    @property
    def fraction(self) -> float | None:
        """Completed fraction of the evaluation budget, when both values are known."""
        budget = self.max_evaluations
        if self.status == "succeeded":
            return 1.0
        if budget is None or budget <= 0 or self.evaluations is None:
            return None
        return min(max(self.evaluations / budget, 0.0), 1.0)


class ProgressMonitor:
    """Incrementally read a job's ``events.jsonl`` and keep a :class:`ProgressState`."""

    def __init__(self, events_path: str | Path) -> None:
        self.path = Path(events_path)
        self.state = ProgressState()
        self._offset = 0

    def poll(self) -> bool:
        """Consume new events; return ``True`` when the state changed."""
        events, self._offset, malformed = read_events(self.path, self._offset)
        self.state.malformed_lines += malformed
        for event in events:
            self.apply(event)
        return bool(events)

    def apply(self, event: Mapping[str, Any]) -> None:
        """Apply one event to the state (unknown event types are ignored)."""
        state = self.state
        elapsed = event.get("time")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
            state.elapsed = float(elapsed)
        kind = event.get("type")
        if kind == "start":
            if not state.finished:
                state.status = "running"
            request = event.get("request")
            if isinstance(request, Mapping):
                state.request = dict(request)
        elif kind == "progress":
            self._apply_progress(event)
        elif kind == "end":
            status = event.get("status")
            state.status = status if isinstance(status, str) and status in TERMINAL_STATUSES else "failed"
            run_dir = event.get("run_dir")
            state.run_dir = run_dir if isinstance(run_dir, str) else None
            error = event.get("error")
            state.error = error if isinstance(error, str) else None
            evaluations = _optional_int(event.get("evaluations"))
            if evaluations is not None:
                state.evaluations = evaluations

    def _apply_progress(self, event: Mapping[str, Any]) -> None:
        state = self.state
        if state.status == "pending":
            state.status = "running"
        state.generation = _optional_int(event.get("generation"))
        state.updates = _optional_int(event.get("update")) or state.updates + 1
        evaluations = _optional_int(event.get("evaluations"))
        if evaluations is not None:
            state.evaluations = evaluations
        front = event.get("front")
        if isinstance(front, list) and front:
            matrix = np.asarray(front, dtype=float)
            if matrix.ndim == 2:
                state.front = matrix
                state.n_obj = int(matrix.shape[1])
        state.front_size = _optional_int(event.get("front_size")) or 0
        value = event.get("hypervolume")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            x = float(state.evaluations) if state.evaluations is not None else float(state.updates)
            state.convergence_x.append(x)
            state.convergence_y.append(float(value))

    def mark_exited(self, returncode: int, *, log_path: str | Path | None = None) -> None:
        """Record a worker that exited without a terminal event."""
        if self.state.finished:
            return
        hint = f" See {log_path}." if log_path is not None else ""
        self.state.status = "failed"
        self.state.error = f"The worker exited with code {returncode} before reporting a result.{hint}"


__all__ = ["ProgressMonitor", "ProgressState"]
