from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from vamos.foundation.observer import RunContext


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class LiveVisualization(Protocol):
    """Callback interface for live/streaming visualization."""

    def on_start(self, ctx: RunContext | None = None) -> None: ...

    def on_generation(
        self,
        generation: int,
        F: NDArray[Any] | None = None,
        X: NDArray[Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None: ...

    def on_end(
        self,
        final_F: NDArray[Any] | None = None,
        final_stats: dict[str, Any] | None = None,
    ) -> None: ...


class NoOpLiveVisualization:
    """Default no-op implementation."""

    def on_start(self, ctx: RunContext | None = None) -> None:
        return None

    def on_generation(
        self,
        generation: int,
        F: NDArray[Any] | None = None,
        X: NDArray[Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        return None

    def on_end(
        self,
        final_F: NDArray[Any] | None = None,
        final_stats: dict[str, Any] | None = None,
    ) -> None:
        return None


def _safe_matplotlib(interactive_backend: str | None = None) -> Any | None:
    try:
        import matplotlib

        if interactive_backend:
            try:
                matplotlib.use(interactive_backend, force=True)
            except Exception:
                _logger().debug("Failed to set matplotlib backend to %r.", interactive_backend, exc_info=True)
        backend = matplotlib.get_backend().lower()
        if "agg" in backend and interactive_backend is None:
            pass
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        _logger().debug("Matplotlib unavailable for live visualization.", exc_info=True)
        return None


@dataclass
class LiveParetoPlot:
    """Live scatter of the population's objective vectors in 2D or 3D.

    Each update rescales the axes to the plotted points. ``objectives_to_plot`` picks the
    objectives (0-based) and sets the number of axes; without it, the first two, or three
    when the problem has three or more objectives. Populations larger than ``max_points``
    are subsampled with a generator seeded by ``rng_seed``.
    """

    update_interval: int = 1
    max_points: int = 1000
    objectives_to_plot: Sequence[int] | None = None
    interactive_backend: str | None = None
    save_final_path: str | os.PathLike[str] | None = None
    title: str = "Population (live)"
    rng_seed: int = 0

    figure: Any = field(default=None, init=False)
    axes: Any = field(default=None, init=False)
    scatter: Any = field(default=None, init=False)
    last_update: int = field(default=-1, init=False)
    dims: int = field(default=2, init=False)
    plt: Any = field(default=None, init=False)
    _interactive: bool = field(default=False, init=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.rng_seed)

    def _plotted_objectives(self, n_obj: int | None = None) -> list[int]:
        if self.objectives_to_plot is not None:
            return list(self.objectives_to_plot)[:3]
        return list(range(self.dims if n_obj is None else min(self.dims, n_obj)))

    def on_start(self, ctx: RunContext | None = None) -> None:
        self.plt = _safe_matplotlib(self.interactive_backend)
        if self.plt is None:
            return
        self.plt.ion()
        backend_name = str(self.plt.get_backend()).lower()
        self._interactive = not any(token in backend_name for token in ("agg", "inline", "pdf", "svg", "ps", "cairo"))
        problem = ctx.problem if ctx is not None else None
        if self.objectives_to_plot is not None:
            self.dims = 3 if len(self.objectives_to_plot) >= 3 else 2
        elif problem is not None and getattr(problem, "n_obj", 2) >= 3:
            self.dims = 3
        self.figure = self.plt.figure(figsize=(6, 5))
        labels = [f"Objective {index + 1}" for index in self._plotted_objectives()]
        if self.dims == 3:
            self.axes = self.figure.add_subplot(111, projection="3d")
            self.axes.set_zlabel(labels[2])
        else:
            self.axes = self.figure.add_subplot(111)
        self.axes.set_xlabel(labels[0])
        self.axes.set_ylabel(labels[1])
        self.axes.set_title(self.title)
        self.figure.tight_layout()
        self.figure.canvas.draw_idle()

    def _tick(self) -> None:
        if self.figure is None:
            return
        self.figure.canvas.draw_idle()
        flush = getattr(self.figure.canvas, "flush_events", None)
        if callable(flush):
            try:
                flush()
            except Exception:
                _logger().debug("Failed to flush matplotlib canvas events", exc_info=True)
        if self._interactive:
            self.plt.pause(0.01)

    def _maybe_subsample(self, F: NDArray[np.float64]) -> NDArray[np.float64]:
        if F.shape[0] <= self.max_points:
            return F
        idx = self._rng.choice(F.shape[0], size=self.max_points, replace=False)
        return F[idx]

    def _fit_limits(self, coords: NDArray[np.float64]) -> None:
        """Rescale the axes to the plotted points, which move as the population converges."""
        if self.dims != 3:
            self.axes.relim()
            self.axes.autoscale_view()
            return
        # 3D scatter offsets are not tracked by relim(): set the limits from the points, with a 5 % margin.
        low = coords.min(axis=0)
        high = coords.max(axis=0)
        span = high - low
        margin = np.where(span > 0.0, 0.05 * span, np.maximum(0.05 * np.abs(low), 0.05))
        self.axes.set_xlim(low[0] - margin[0], high[0] + margin[0])
        self.axes.set_ylim(low[1] - margin[1], high[1] + margin[1])
        self.axes.set_zlim(low[2] - margin[2], high[2] + margin[2])

    def on_generation(
        self,
        generation: int,
        F: NDArray[Any] | None = None,
        X: NDArray[Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        if self.plt is None or F is None or generation - self.last_update < self.update_interval:
            return
        self.last_update = generation
        data = self._maybe_subsample(np.asarray(F, dtype=float))
        if data.ndim != 2 or data.shape[0] == 0:
            return
        dims = self.dims
        idx = self._plotted_objectives(data.shape[1])
        coords = data[:, idx]
        if self.scatter is None:
            if dims == 3:
                self.scatter = self.axes.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=18, alpha=0.8)
            else:
                (self.scatter,) = self.axes.plot(coords[:, 0], coords[:, 1], "o", markersize=4, alpha=0.8)
        else:
            if dims == 3:
                self.scatter._offsets3d = (coords[:, 0], coords[:, 1], coords[:, 2])
            else:
                self.scatter.set_data(coords[:, 0], coords[:, 1])
        self._fit_limits(coords)
        if stats and "hv" in stats:
            self.axes.set_title(f"{self.title} | HV={stats['hv']:.4f}")
        self._tick()

    def on_end(
        self,
        final_F: NDArray[Any] | None = None,
        final_stats: dict[str, Any] | None = None,
    ) -> None:
        if self.plt is None:
            return
        self.on_generation(self.last_update + 1, final_F, stats=final_stats)
        if self.save_final_path:
            path = os.fspath(self.save_final_path)
            dir_name = os.path.dirname(path) or "."
            os.makedirs(dir_name, exist_ok=True)
            self.figure.savefig(path, dpi=200)


@dataclass
class LiveTuningPlot:
    """Simple live scatter for tuning/meta-optimization results."""

    x_param: str
    y_metric: str = "hv"
    color_param: str | None = None
    update_interval: int = 1
    save_final_path: str | os.PathLike[str] | None = None
    interactive_backend: str | None = None

    plt: Any = field(default=None, init=False)
    figure: Any = field(default=None, init=False)
    axes: Any = field(default=None, init=False)
    last_update: int = field(default=-1, init=False)
    _interactive: bool = field(default=False, init=False)

    def on_start(self, ctx: RunContext | None = None) -> None:
        self.plt = _safe_matplotlib(self.interactive_backend)
        if self.plt is None:
            return
        self.plt.ion()
        backend_name = str(self.plt.get_backend()).lower()
        self._interactive = not any(token in backend_name for token in ("agg", "inline", "pdf", "svg", "ps", "cairo"))
        self.figure, self.axes = self.plt.subplots(figsize=(6, 4))
        self.axes.set_xlabel(self.x_param)
        self.axes.set_ylabel(self.y_metric)
        self.figure.tight_layout()

    def _tick(self) -> None:
        if self.figure is None:
            return
        self.figure.canvas.draw_idle()
        flush = getattr(self.figure.canvas, "flush_events", None)
        if callable(flush):
            try:
                flush()
            except Exception:
                _logger().debug("Failed to flush matplotlib canvas events", exc_info=True)
        if self._interactive:
            self.plt.pause(0.01)

    def on_generation(
        self,
        generation: int,
        F: NDArray[Any] | None = None,
        X: NDArray[Any] | None = None,
        stats: dict[str, Any] | None = None,
    ) -> None:
        if self.plt is None or stats is None:
            return
        if generation - self.last_update < self.update_interval:
            return
        self.last_update = generation
        x_vals = stats.get("x_vals")
        y_vals = stats.get("y_vals")
        colors = stats.get("colors")
        if x_vals is None or y_vals is None:
            return
        self.axes.clear()
        self.axes.set_xlabel(self.x_param)
        self.axes.set_ylabel(self.y_metric)
        self.axes.scatter(x_vals, y_vals, c=colors if colors is not None else "tab:blue", s=25, alpha=0.8)
        self._tick()

    def on_end(
        self,
        final_F: NDArray[Any] | None = None,
        final_stats: dict[str, Any] | None = None,
    ) -> None:
        if self.plt is None:
            return
        if self.save_final_path:
            path = os.fspath(self.save_final_path)
            dir_name = os.path.dirname(path) or "."
            os.makedirs(dir_name, exist_ok=True)
            self.figure.savefig(path, dpi=200)
