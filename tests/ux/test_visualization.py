from types import SimpleNamespace

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from vamos.ux.visualization import (
    plot_hv_convergence,
    plot_parallel_coordinates,
    plot_pareto_front_2d,
    plot_pareto_front_3d,
)


def test_plot_pareto_front_2d_smoke():
    F = np.array([[1.0, 2.0], [0.5, 1.5]])
    ax = plot_pareto_front_2d(F, show=False)
    assert hasattr(ax, "scatter")


def test_plot_pareto_front_3d_smoke():
    F = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    ax = plot_pareto_front_3d(F, show=False)
    assert hasattr(ax, "scatter")


def _ascending(limits):
    return limits[0] < limits[1]


def test_pareto_front_plots_do_not_invert_axes():
    # Objectives are minimized: the ideal corner is at the lower left, as in every other VAMOS front plot.
    ax2 = plot_pareto_front_2d(np.array([[0.2, 0.9], [0.5, 0.5], [0.9, 0.1]]))
    ax3 = plot_pareto_front_3d(np.array([[0.2, 0.9, 0.4], [0.5, 0.5, 0.1], [0.9, 0.1, 0.7]]))

    assert _ascending(ax2.get_xlim()) and _ascending(ax2.get_ylim())
    assert _ascending(ax3.get_xlim()) and _ascending(ax3.get_ylim()) and _ascending(ax3.get_zlim())


def test_pareto_front_plots_invert_axes_on_request():
    ax2 = plot_pareto_front_2d(np.array([[0.2, 0.9], [0.9, 0.1]]), invert_axes=True)
    ax3 = plot_pareto_front_3d(np.array([[0.2, 0.9, 0.4], [0.9, 0.1, 0.7]]), invert_axes=True)

    assert not _ascending(ax2.get_xlim()) and not _ascending(ax2.get_ylim())
    assert not _ascending(ax3.get_zlim())


def test_plot_parallel_coordinates_smoke():
    F = np.array([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    ax = plot_parallel_coordinates(F, show=False)
    assert hasattr(ax, "plot")


def test_plot_hv_convergence_smoke():
    evals = np.array([0, 10, 20])
    hv = np.array([0.1, 0.2, 0.3])
    ax = plot_hv_convergence(evals, hv, show=False)
    assert hasattr(ax, "plot")


def test_explore_result_front_hover_matches_solution():
    pytest.importorskip("plotly")
    from vamos.ux.results import explore_result_front

    # The second solution is dominated; each X row identifies its solution.
    F = np.array([[0.2, 0.6], [0.5, 0.5], [0.4, 0.4], [0.6, 0.2]])
    X = np.array([[0.0], [111.0], [222.0], [333.0]])
    result = SimpleNamespace(F=F, X=X, G=None, meta={})

    fig = explore_result_front(result)

    texts = {(round(x, 4), round(y, 4)): text for x, y, text in zip(fig.data[0].x, fig.data[0].y, fig.data[0].hovertext, strict=True)}
    assert "x1: 222.0000" in texts[(0.4, 0.4)]
    assert len(texts) == 3


def test_explore_result_front_keeps_only_feasible_solutions():
    pytest.importorskip("plotly")
    from vamos.ux.results import explore_result_front

    F = np.array([[0.1, 0.1], [0.4, 0.4], [0.6, 0.2]])
    X = np.array([[1.0], [2.0], [3.0]])
    G = np.array([[1.0], [0.0], [-1.0]])  # the first solution dominates the others but is infeasible
    result = SimpleNamespace(F=F, X=X, G=G, meta={})

    fig = explore_result_front(result)

    assert sorted(zip(fig.data[0].x, fig.data[0].y, strict=True)) == [(0.4, 0.4), (0.6, 0.2)]


def test_explore_result_front_reads_constraints_from_an_optimization_result():
    pytest.importorskip("plotly")
    from vamos.experiment.optimization_result.model import OptimizationResult
    from vamos.ux.results import explore_result_front

    # OptimizationResult keeps constraint values in its payload, not in a G attribute.
    F = np.array([[0.1, 0.1], [0.4, 0.4], [0.6, 0.2]])
    result = OptimizationResult({"F": F, "X": np.array([[1.0], [2.0], [3.0]]), "G": np.array([[1.0], [0.0], [-1.0]])})

    fig = explore_result_front(result)

    shown = sorted(zip(fig.data[0].x, fig.data[0].y, strict=True))
    assert shown == [(0.4, 0.4), (0.6, 0.2)]
    assert shown == sorted(map(tuple, np.asarray(result.front())))
