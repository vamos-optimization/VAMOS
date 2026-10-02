"""Progress events written by GUI-launched runs and read by the interface.

A worker appends JSON Lines to ``events.jsonl`` in its job directory: one
``start`` event, time-throttled ``progress`` events, and one terminal ``end``
event. Readers consume complete lines only, so a partially written last line is
picked up by the next poll, and malformed lines are counted and skipped.

Progress events carry the current non-dominated front (finite rows only,
deterministically down-sampled for display) and a progress hypervolume computed
against a reference point fixed from the first reported population. That value
tracks the progress of one run; it is not comparable across runs.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from vamos.foundation.quality_indicators.hypervolume import hypervolume
from vamos.foundation.quality_indicators.moocore_indicators import has_moocore
from vamos.foundation.quality_indicators.pareto import pareto_filter

if TYPE_CHECKING:
    from vamos.foundation.observer import RunContext

EVENTS_FILENAME = "events.jsonl"
EVENT_SCHEMA = "vamos.gui-events"
EVENT_SCHEMA_VERSION = 1
TERMINAL_STATUSES = ("succeeded", "failed", "cancelled")
DEFAULT_MIN_INTERVAL = 0.25
DEFAULT_MAX_POINTS = 500
_HV_MAX_OBJECTIVES_FALLBACK = 3
_HV_MAX_OBJECTIVES_MOOCORE = 8
_SIGNIFICANT_DIGITS = 6
_EVALUATION_KEYS = ("evals", "evaluations", "n_eval")

FloatMatrix = npt.NDArray[np.float64]


def finite_rows(F: npt.ArrayLike) -> FloatMatrix:
    """Return ``F`` as a float matrix without rows that contain NaN or infinity."""
    arr = np.asarray(F, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        return np.empty((0, 0), dtype=float)
    return np.asarray(arr[np.all(np.isfinite(arr), axis=1)], dtype=np.float64)


def nondominated(F: FloatMatrix) -> FloatMatrix:
    """Return the non-dominated rows of a finite objective matrix (minimization)."""
    if F.shape[0] == 0:
        return F
    front, _ = pareto_filter(F, return_indices=True)
    return np.asarray(front, dtype=float)


def downsample(front: FloatMatrix, max_points: int) -> FloatMatrix:
    """Keep at most ``max_points`` rows, evenly spaced along the first objective."""
    if max_points < 1:
        raise ValueError(f"max_points must be positive, got {max_points}.")
    size = front.shape[0]
    if size <= max_points:
        return front
    order = np.argsort(front[:, 0], kind="mergesort")
    keep = np.unique(np.linspace(0, size - 1, max_points).round().astype(int))
    return np.asarray(front[order[keep]], dtype=np.float64)


class ProgressHypervolume:
    """Hypervolume against a reference point fixed from the first population."""

    def __init__(self) -> None:
        self.reference: FloatMatrix | None = None

    @staticmethod
    def supported(n_obj: int) -> bool:
        """Return whether a cheap enough hypervolume is available for ``n_obj``."""
        if n_obj <= _HV_MAX_OBJECTIVES_FALLBACK:
            return True
        return n_obj <= _HV_MAX_OBJECTIVES_MOOCORE and has_moocore()

    def __call__(self, population: FloatMatrix, front: FloatMatrix) -> float | None:
        if front.shape[0] == 0 or not self.supported(front.shape[1]):
            return None
        if self.reference is None:
            worst = population.max(axis=0)
            span = worst - population.min(axis=0)
            margin = np.where(span > 0.0, 0.1 * span, np.maximum(0.1 * np.abs(worst), 1.0))
            self.reference = worst + margin
        reference = self.reference
        if reference.shape[0] != front.shape[1]:
            return None
        inside = front[np.all(front < reference, axis=1)]
        if inside.shape[0] == 0:
            return 0.0
        return float(hypervolume(inside, reference))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _rounded(matrix: FloatMatrix) -> list[list[float]]:
    return [[float(f"{value:.{_SIGNIFICANT_DIGITS}g}") for value in row] for row in matrix.tolist()]


def _evaluations(stats: Mapping[str, Any] | None) -> int | None:
    if not isinstance(stats, Mapping):
        return None
    for key in _EVALUATION_KEYS:
        value = stats.get(key)
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, (float, np.floating)) and math.isfinite(float(value)) and float(value).is_integer():
            return int(value)
    return None


def append_event(path: str | Path, event: Mapping[str, Any]) -> None:
    """Append ``event`` as one JSON line, starting a new line if the file ends mid-line."""
    target = Path(path)
    line = json.dumps(_jsonable(event), allow_nan=False, separators=(",", ":"))
    prefix = ""
    if target.exists() and target.stat().st_size > 0:
        with target.open("rb") as handle:
            handle.seek(-1, 2)
            if handle.read(1) != b"\n":
                prefix = "\n"
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"{prefix}{line}\n")


def read_events(path: str | Path, offset: int = 0) -> tuple[list[dict[str, Any]], int, int]:
    """Read complete event lines after byte ``offset``.

    Returns ``(events, new_offset, malformed_lines)``. An incomplete trailing line
    is left in place for the next call; a missing file yields no events.
    """
    target = Path(path)
    try:
        with target.open("rb") as handle:
            handle.seek(offset)
            data = handle.read()
    except FileNotFoundError:
        return [], offset, 0
    end = data.rfind(b"\n")
    if end < 0:
        return [], offset, 0
    events: list[dict[str, Any]] = []
    malformed = 0
    for raw in data[: end + 1].splitlines():
        if not raw.strip():
            continue
        try:
            item = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            malformed += 1
            continue
        if isinstance(item, dict) and isinstance(item.get("type"), str):
            events.append(item)
        else:
            malformed += 1
    return events, offset + end + 1, malformed


def has_terminal_event(path: str | Path) -> bool:
    """Return whether ``path`` already contains an ``end`` event."""
    events, _, _ = read_events(path)
    return any(event.get("type") == "end" for event in events)


class ProgressEventWriter:
    """Live-visualization hook that appends throttled progress events to a JSONL file.

    It implements the ``on_start``/``on_generation``/``on_end`` protocol accepted by
    ``optimize(..., live_viz=...)``. Generations arriving faster than
    ``min_interval`` seconds are coalesced; ``on_end`` always flushes the latest one.
    The terminal ``end`` event is written separately with :meth:`write_end` once the
    run has been stored, so it can reference the stored run directory.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        request: Mapping[str, Any] | None = None,
        min_interval: float = DEFAULT_MIN_INTERVAL,
        max_points: int = DEFAULT_MAX_POINTS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if min_interval < 0.0:
            raise ValueError(f"min_interval must be non-negative, got {min_interval}.")
        if max_points < 1:
            raise ValueError(f"max_points must be positive, got {max_points}.")
        self.path = Path(path)
        self._request = dict(request or {})
        self._min_interval = float(min_interval)
        self._max_points = int(max_points)
        self._clock = clock
        self._started_at: float | None = None
        self._last_emit: float | None = None
        self._updates = 0
        self._pending: tuple[int, FloatMatrix, int | None] | None = None
        self._hypervolume = ProgressHypervolume()
        self._ended = False

    def on_start(self, ctx: RunContext | None = None) -> None:
        if self._started_at is not None:
            return
        self._started_at = self._clock()
        self._append({"type": "start", "schema": EVENT_SCHEMA, "version": EVENT_SCHEMA_VERSION, "request": self._request})

    def on_generation(
        self,
        generation: int,
        F: npt.NDArray[Any] | None = None,
        X: npt.NDArray[Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        self.on_start()
        self._updates += 1
        if F is None:
            return
        evaluations = _evaluations(stats)
        now = self._clock()
        if self._last_emit is not None and now - self._last_emit < self._min_interval:
            self._pending = (int(generation), np.array(F, dtype=float, copy=True), evaluations)
            return
        self._emit_progress(int(generation), np.asarray(F, dtype=float), evaluations, now)

    def on_end(
        self,
        final_F: npt.NDArray[Any] | None = None,
        final_stats: dict[str, Any] | None = None,
    ) -> None:
        self.on_start()
        if self._pending is not None:
            generation, F, evaluations = self._pending
            self._emit_progress(generation, F, evaluations, self._clock())
        elif self._last_emit is None and final_F is not None:
            self._emit_progress(self._updates, np.asarray(final_F, dtype=float), _evaluations(final_stats), self._clock())

    def write_end(self, status: str, *, run_dir: str | None = None, evaluations: int | None = None, error: str | None = None) -> None:
        """Write the terminal event once; later calls are ignored."""
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"Unknown terminal status {status!r}; expected one of {TERMINAL_STATUSES}.")
        if self._ended:
            return
        self.on_start()
        self._ended = True
        self._append({"type": "end", "status": status, "run_dir": run_dir, "evaluations": evaluations, "error": error})

    def _emit_progress(self, generation: int, F: FloatMatrix, evaluations: int | None, now: float) -> None:
        self._pending = None
        self._last_emit = now
        population = finite_rows(F)
        front = nondominated(population)
        shown = downsample(front, self._max_points) if front.shape[0] else front
        hv = self._hypervolume(population, front) if front.shape[0] else None
        self._append(
            {
                "type": "progress",
                "generation": generation,
                "update": self._updates,
                "evaluations": evaluations,
                "n_obj": int(front.shape[1]) if front.shape[0] else None,
                "front_size": int(front.shape[0]),
                "front": _rounded(shown),
                "hypervolume": hv,
            }
        )

    def _append(self, event: Mapping[str, Any]) -> None:
        started = self._started_at if self._started_at is not None else self._clock()
        append_event(self.path, {"time": round(self._clock() - started, 3), **event})


__all__ = [
    "DEFAULT_MAX_POINTS",
    "DEFAULT_MIN_INTERVAL",
    "EVENTS_FILENAME",
    "EVENT_SCHEMA",
    "EVENT_SCHEMA_VERSION",
    "TERMINAL_STATUSES",
    "ProgressEventWriter",
    "ProgressHypervolume",
    "append_event",
    "downsample",
    "finite_rows",
    "has_terminal_event",
    "nondominated",
    "read_events",
]
