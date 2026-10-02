from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("plotly")

from vamos.ux.visualization.interactive import (
    CONVERGENCE_UIREVISION,
    FRONT_UIREVISION,
    WEBGL_THRESHOLD,
    convergence_figure,
    front_figure,
    objective_labels,
    placeholder_figure,
)

pytestmark = pytest.mark.gui


def _objectives(n_points: int, n_obj: int) -> np.ndarray:
    return np.random.default_rng(0).random((n_points, n_obj))


def test_two_objective_view_supports_selection_and_stable_ui_state() -> None:
    fig = front_figure(_objectives(50, 2))
    trace = fig.data[0]
    assert trace.type == "scatter"
    assert list(trace.customdata) == list(range(50))
    assert fig.layout.uirevision == FRONT_UIREVISION
    assert fig.layout.dragmode == "lasso"
    assert (fig.layout.xaxis.title.text, fig.layout.yaxis.title.text) == ("f1", "f2")


def test_large_fronts_use_webgl() -> None:
    assert front_figure(_objectives(WEBGL_THRESHOLD + 1, 2)).data[0].type == "scattergl"


def test_projection_axes_names_and_custom_indices() -> None:
    F = _objectives(10, 4)
    fig = front_figure(F, axes=(2, 3), indices=range(100, 110), objective_names=["a", "b", "c", "d"], uirevision="fixed")
    trace = fig.data[0]
    np.testing.assert_allclose(trace.x, F[:, 2])
    np.testing.assert_allclose(trace.y, F[:, 3])
    assert list(trace.customdata) == list(range(100, 110))
    assert fig.layout.xaxis.title.text == "c"
    assert fig.layout.uirevision == "fixed"


def test_three_dimensional_and_parallel_views() -> None:
    F = _objectives(20, 3)
    scatter = front_figure(F, view="3d").data[0]
    assert scatter.type == "scatter3d"
    assert list(scatter.customdata) == list(range(20))
    parallel = front_figure(F, view="parallel").data[0]
    assert parallel.type == "parcoords"
    assert [dimension.label for dimension in parallel.dimensions] == ["f1", "f2", "f3"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"view": "3d"}, "at least three objectives"),
        ({"axes": (0, 0)}, "distinct"),
        ({"axes": (0, 5)}, "out of range"),
        ({"axes": (0,)}, "Expected 2 objective axes"),
        ({"view": "polar"}, "Unknown front view"),
        ({"indices": [1, 2]}, "Expected 5 indices"),
    ],
)
def test_invalid_figure_requests_fail_clearly(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        front_figure(_objectives(5, 2), **kwargs)  # type: ignore[arg-type]


def test_objective_matrix_shape_is_validated() -> None:
    with pytest.raises(ValueError, match="objective matrix"):
        front_figure(np.zeros(5))


def test_convergence_and_placeholder_figures() -> None:
    fig = convergence_figure([100, 200], [0.5, 0.7], label="HV")
    assert list(fig.data[0].x) == [100.0, 200.0]
    assert fig.layout.uirevision == CONVERGENCE_UIREVISION
    assert fig.layout.yaxis.title.text == "HV"
    with pytest.raises(ValueError, match="equal length"):
        convergence_figure([1.0], [1.0, 2.0])
    assert placeholder_figure("waiting").layout.annotations[0].text == "waiting"


def test_objective_labels() -> None:
    assert objective_labels(3) == ["f1", "f2", "f3"]
    assert objective_labels(2, ["cost", "risk"]) == ["cost", "risk"]
    with pytest.raises(ValueError, match="Expected 2 objective names"):
        objective_labels(2, ["cost"])
