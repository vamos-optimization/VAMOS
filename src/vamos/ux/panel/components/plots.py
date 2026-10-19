"""Plotly figures shared by the Studio pages."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def objective_names(n_obj: int) -> list[str]:
    return [f"f{index + 1}" for index in range(n_obj)]


def front_figure(F: Any, *, title: str, height: int = 450) -> Any:
    """Scatter one front in 2D or 3D, or as parallel coordinates beyond three objectives."""
    import pandas as pd
    import plotly.express as px

    points = np.atleast_2d(np.asarray(F, dtype=float))
    names = objective_names(points.shape[1])
    frame = pd.DataFrame(points, columns=names)
    if points.shape[1] == 1:
        fig = px.scatter(frame, x="f1", y=np.zeros(points.shape[0]), labels={"y": ""})
    elif points.shape[1] == 2:
        fig = px.scatter(frame, x="f1", y="f2")
    elif points.shape[1] == 3:
        fig = px.scatter_3d(frame, x="f1", y="f2", z="f3")
    else:
        fig = px.parallel_coordinates(frame, dimensions=names)
    fig.update_layout(title=title, height=height)
    return fig


def overlay_figure(fronts: Sequence[tuple[str, Any]], *, title: str, height: int = 480) -> Any:
    """Overlay the fronts of several algorithms on one problem, coloured by algorithm."""
    import pandas as pd
    import plotly.express as px

    n_obj = min(np.atleast_2d(np.asarray(F)).shape[1] for _, F in fronts)
    names = objective_names(n_obj)
    frames = []
    for label, F in fronts:
        points = np.atleast_2d(np.asarray(F, dtype=float))[:, :n_obj]
        frame = pd.DataFrame(points, columns=names)
        frame.insert(0, "Algorithm", label)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if n_obj == 2:
        fig = px.scatter(data, x="f1", y="f2", color="Algorithm")
    elif n_obj == 3:
        fig = px.scatter_3d(data, x="f1", y="f2", z="f3", color="Algorithm")
    else:
        fig = px.scatter_matrix(data, dimensions=names, color="Algorithm")
    fig.update_layout(title=title, height=height)
    return fig


def a12_heatmap(problems: Sequence[str], contenders: Sequence[str], effects: Any, labels: Any, *, pivot: str) -> Any:
    """A12 of every contender against the pivot; 0.5 is no effect, above 0.5 the contender is better."""
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Heatmap(
            z=effects,
            x=list(contenders),
            y=list(problems),
            zmin=0.0,
            zmax=1.0,
            zmid=0.5,
            colorscale="RdBu",
            text=labels,
            texttemplate="%{text}",
            colorbar={"title": "A12"},
            hovertemplate="%{y} · %{x}: A12 %{z:.2f}<extra></extra>",
        )
    )
    fig.update_traces(textfont={"size": 13})
    fig.update_layout(title=f"Vargha-Delaney A12 against {pivot} (above 0.5: better than {pivot})", height=180 + 70 * len(problems))
    return fig


def cd_figure(algorithms: Sequence[str], ranks: Sequence[float], cd: float | None, groups: Sequence[tuple[int, int]]) -> Any:
    """Demsar's critical-difference diagram: average ranks, with bars joining algorithms that do not differ."""
    import plotly.graph_objects as go

    order = np.argsort(ranks)
    names = [algorithms[i] for i in order]
    values = [float(ranks[i]) for i in order]
    k = len(values)
    fig = go.Figure()
    fig.add_shape(type="line", x0=1, x1=max(k, 2), y0=0, y1=0, line={"color": "black"})
    for tick in range(1, max(k, 2) + 1):
        fig.add_shape(type="line", x0=tick, x1=tick, y0=0, y1=0.08, line={"color": "black"})
    for position, (name, value) in enumerate(zip(names, values, strict=True)):
        depth = -0.35 * (position + 1)
        fig.add_shape(type="line", x0=value, x1=value, y0=0, y1=depth, line={"color": "#555"})
        fig.add_annotation(x=value, y=depth, text=f"{name} ({value:.2f})", showarrow=False, yshift=-10)
    for level, (start, end) in enumerate(groups):
        y = 0.25 + 0.12 * level
        fig.add_shape(type="line", x0=values[start] - 0.03, x1=values[end] + 0.03, y0=y, y1=y, line={"color": "#d62728", "width": 5})
    if cd is not None:
        top = 0.35 + 0.12 * len(groups)
        fig.add_shape(type="line", x0=1, x1=1 + cd, y0=top, y1=top, line={"color": "black", "width": 2})
        fig.add_annotation(x=1 + cd / 2, y=top, text=f"CD = {cd:.2f}", showarrow=False, yshift=12)
    fig.update_xaxes(range=[0.6, max(k, 2) + 0.4], title="Average rank (1 is best)", zeroline=False)
    fig.update_yaxes(visible=False, range=[-0.35 * (k + 1) - 0.2, 0.6 + 0.12 * len(groups)])
    fig.update_layout(title="Critical difference (Nemenyi, α = 0.05)", height=220 + 40 * k, showlegend=False, plot_bgcolor="white")
    return fig


def box_figure(rows: Sequence[dict[str, object]], *, indicator: str) -> Any:
    """The distribution of the indicator over the runs, one panel per problem with its own scale."""
    import pandas as pd
    import plotly.express as px

    frame = pd.DataFrame(rows)
    problems = frame["Problem"].nunique() if not frame.empty else 1
    fig = px.box(frame, x="Algorithm", y=indicator, color="Algorithm", facet_col="Problem", facet_col_wrap=3, points="all")
    fig.update_yaxes(matches=None, showticklabels=True)
    fig.update_layout(height=320 * int(np.ceil(problems / 3)), showlegend=False, title=f"{indicator} of every run")
    return fig


def attainment_figure(surfaces: Sequence[tuple[str, str, Any]], *, title: str) -> Any:
    """Attainment surfaces as staircases: ``(algorithm, level label, points)`` per line."""
    import plotly.graph_objects as go

    fig = go.Figure()
    dashes = {"best": "dot", "median": "solid", "worst": "dash"}
    for algorithm, level, points in surfaces:
        pts = np.asarray(points, dtype=float)
        if pts.size == 0:
            continue
        fig.add_trace(
            go.Scatter(
                x=pts[:, 0],
                y=pts[:, 1],
                mode="lines",
                line_shape="hv",
                name=f"{algorithm} · {level}",
                line={"dash": dashes.get(level, "solid")},
            )
        )
    fig.update_layout(title=title, xaxis_title="f1", yaxis_title="f2", height=480)
    return fig


def cost_figure(rows: Sequence[dict[str, object]]) -> Any:
    import pandas as pd
    import plotly.express as px

    fig = px.bar(pd.DataFrame(rows), x="Problem", y="Median runtime (s)", color="Algorithm", barmode="group")
    fig.update_layout(title="Median runtime per run", height=420)
    return fig


__all__ = [
    "a12_heatmap",
    "attainment_figure",
    "box_figure",
    "cd_figure",
    "cost_figure",
    "front_figure",
    "objective_names",
    "overlay_figure",
]
