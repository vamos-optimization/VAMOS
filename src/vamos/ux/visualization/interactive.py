"""Plotly figure builders for interactive front exploration and live progress.

The builders are pure functions: they receive arrays and return
``plotly.graph_objects.Figure`` instances, so the same figures serve the GUI,
notebooks, and exported HTML. Plotly is imported lazily; importing this module
does not require it.

Every figure sets a constant ``layout.uirevision`` so that zoom, camera, and
selection survive repeated updates while a run is still progressing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

FrontView = Literal["2d", "3d", "parallel"]

WEBGL_THRESHOLD = 1500
FRONT_UIREVISION = "vamos-front"
CONVERGENCE_UIREVISION = "vamos-convergence"
_INSTALL_HINT = 'Interactive figures require plotly. Install it with: pip install "vamos-optimization[gui]"'


def _require_plotly() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - exercised only without plotly
        raise ImportError(_INSTALL_HINT) from exc
    return go


def objective_labels(n_obj: int, names: Sequence[str] | None = None) -> list[str]:
    """Return display labels ``f1..fM`` or validated custom names."""
    if n_obj < 1:
        raise ValueError(f"n_obj must be positive, got {n_obj}.")
    if names is None:
        return [f"f{i + 1}" for i in range(n_obj)]
    labels = [str(name) for name in names]
    if len(labels) != n_obj:
        raise ValueError(f"Expected {n_obj} objective names, got {len(labels)}.")
    return labels


def _as_objectives(F: npt.ArrayLike) -> npt.NDArray[np.float64]:
    arr = np.asarray(F, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected an objective matrix of shape (N, n_obj >= 2), got shape {arr.shape}.")
    return arr


def _resolve_axes(axes: Sequence[int] | None, n_obj: int, count: int) -> tuple[int, ...]:
    resolved = tuple(range(count)) if axes is None else tuple(int(axis) for axis in axes)
    if len(resolved) != count:
        raise ValueError(f"Expected {count} objective axes, got {len(resolved)}.")
    if len(set(resolved)) != count:
        raise ValueError(f"Objective axes must be distinct, got {resolved}.")
    for axis in resolved:
        if not 0 <= axis < n_obj:
            raise ValueError(f"Objective axis {axis} is out of range for {n_obj} objectives.")
    return resolved


def _base_layout(fig: Any, *, title: str | None, uirevision: str) -> None:
    fig.update_layout(
        title=title,
        uirevision=uirevision,
        template="plotly_white",
        showlegend=False,
        margin={"l": 50, "r": 20, "t": 50 if title else 20, "b": 50},
    )


def front_figure(
    F: npt.ArrayLike,
    *,
    view: FrontView = "2d",
    axes: Sequence[int] | None = None,
    objective_names: Sequence[str] | None = None,
    indices: Sequence[int] | None = None,
    title: str | None = None,
    uirevision: str = FRONT_UIREVISION,
) -> Any:
    """Return a Plotly figure of an objective matrix ``F`` with shape ``(N, M)``.

    ``view`` is ``"2d"`` (a projection on two objectives, lasso selection by
    default), ``"3d"`` (three objectives), or ``"parallel"`` (all objectives).
    ``indices`` identify the rows of ``F`` and are attached as ``customdata``
    so selection and click events can be mapped back to solutions; they
    default to ``range(N)``.
    """
    go = _require_plotly()
    arr = _as_objectives(F)
    n_points, n_obj = arr.shape
    labels = objective_labels(n_obj, objective_names)
    ids = np.arange(n_points) if indices is None else np.asarray(indices, dtype=int)
    if ids.shape != (n_points,):
        raise ValueError(f"Expected {n_points} indices, got shape {ids.shape}.")
    fig = go.Figure()
    if view == "2d":
        i, j = _resolve_axes(axes, n_obj, 2)
        trace_type = go.Scattergl if n_points > WEBGL_THRESHOLD else go.Scatter
        fig.add_trace(
            trace_type(
                x=arr[:, i],
                y=arr[:, j],
                mode="markers",
                marker={"size": 7, "opacity": 0.85},
                customdata=ids,
                hovertemplate=f"#%{{customdata}}<br>{labels[i]}=%{{x:.6g}}<br>{labels[j]}=%{{y:.6g}}<extra></extra>",
            )
        )
        fig.update_layout(xaxis_title=labels[i], yaxis_title=labels[j], dragmode="lasso")
    elif view == "3d":
        if n_obj < 3:
            raise ValueError("A 3-D view requires at least three objectives.")
        i, j, k = _resolve_axes(axes, n_obj, 3)
        fig.add_trace(
            go.Scatter3d(
                x=arr[:, i],
                y=arr[:, j],
                z=arr[:, k],
                mode="markers",
                marker={"size": 3, "opacity": 0.85},
                customdata=ids,
                hovertemplate=(
                    f"#%{{customdata}}<br>{labels[i]}=%{{x:.6g}}<br>{labels[j]}=%{{y:.6g}}<br>{labels[k]}=%{{z:.6g}}<extra></extra>"
                ),
            )
        )
        fig.update_layout(scene={"xaxis_title": labels[i], "yaxis_title": labels[j], "zaxis_title": labels[k]})
    elif view == "parallel":
        fig.add_trace(
            go.Parcoords(
                line={"color": arr[:, 0], "colorscale": "Viridis"},
                dimensions=[{"label": labels[m], "values": arr[:, m]} for m in range(n_obj)],
            )
        )
    else:
        raise ValueError(f"Unknown front view {view!r}; expected '2d', '3d', or 'parallel'.")
    _base_layout(fig, title=title, uirevision=uirevision)
    return fig


def convergence_figure(
    evaluations: Sequence[float],
    values: Sequence[float],
    *,
    label: str = "Hypervolume",
    title: str | None = None,
    uirevision: str = CONVERGENCE_UIREVISION,
) -> Any:
    """Return a line chart of an indicator against evaluations."""
    go = _require_plotly()
    x = np.asarray(evaluations, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"evaluations and values must be 1-D sequences of equal length, got {x.shape} and {y.shape}.")
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers", marker={"size": 4}, hovertemplate="%{x:.0f}: %{y:.6g}<extra></extra>"))
    fig.update_layout(xaxis_title="Evaluations", yaxis_title=label)
    _base_layout(fig, title=title, uirevision=uirevision)
    return fig


def placeholder_figure(message: str, *, uirevision: str = FRONT_UIREVISION) -> Any:
    """Return an empty figure that displays ``message`` (used before data exists)."""
    go = _require_plotly()
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    _base_layout(fig, title=None, uirevision=uirevision)
    return fig


__all__ = [
    "CONVERGENCE_UIREVISION",
    "FRONT_UIREVISION",
    "WEBGL_THRESHOLD",
    "FrontView",
    "convergence_figure",
    "front_figure",
    "objective_labels",
    "placeholder_figure",
]
