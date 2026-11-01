import numpy as np
import pytest

pytest.importorskip("matplotlib")

from vamos.engine.algorithm.agemoea import AGEMOEA
from vamos.engine.algorithm.config import AGEMOEAConfig, NSGAIIConfig, RVEAConfig
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.engine.algorithm.rvea import RVEA
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.dtlz import DTLZ2Problem
from vamos.foundation.problem.zdt1 import ZDT1Problem
from vamos.ux.visualization.live_viz import LiveParetoPlot, LiveVisualization


class DummyProblem:
    def __init__(self):
        self.n_var = 2
        self.n_obj = 2
        self.xl = 0.0
        self.xu = 1.0
        self.encoding = "real"

    def evaluate(self, X, out):
        f1 = X[:, 0]
        f2 = 1.0 - X[:, 1]
        out["F"] = np.stack([f1, f2], axis=1)


class RecorderViz(LiveVisualization):
    def __init__(self):
        self.starts = 0
        self.gens = 0
        self.ends = 0

    def on_start(self, problem=None, algorithm=None, config=None):
        self.starts += 1

    def on_generation(self, generation: int, F=None, X=None, stats=None):
        self.gens += 1

    def on_end(self, final_F=None, final_stats=None):
        self.ends += 1


def test_live_viz_callbacks_invoked(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(6)
        .offspring_size(6)
        .crossover("sbx", prob=0.9, eta=10.0)
        .mutation("polynomial", prob="1/n", eta=10.0)
        .selection("tournament", size=2)
        .result_mode("population")
        .build()
    )
    algo = NSGAII(cfg.to_dict(), kernel=NumPyKernel())
    problem = DummyProblem()
    recorder = RecorderViz()

    result = algo.run(problem, termination=("max_evaluations", 12), seed=1, live_viz=recorder)

    assert result["F"].shape[0] == cfg.pop_size
    assert recorder.starts == 1
    assert recorder.ends == 1
    assert recorder.gens >= 1


def test_agemoea_live_viz_callbacks_invoked(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    cfg = AGEMOEAConfig.builder().pop_size(8).crossover("sbx", prob=0.9, eta=10.0).mutation("polynomial", prob=0.25, eta=10.0).build()
    algo = AGEMOEA(cfg.to_dict(), kernel=NumPyKernel())
    recorder = RecorderViz()

    result = algo.run(ZDT1Problem(n_var=4), termination=("max_evaluations", 16), seed=2, live_viz=recorder)

    assert result["F"].shape[0] > 0
    assert recorder.starts == 1
    assert recorder.ends == 1
    assert recorder.gens >= 1


def test_rvea_live_viz_callbacks_invoked(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    cfg = (
        RVEAConfig.builder()
        .pop_size(10)
        .n_partitions(3)
        .alpha(2.0)
        .adapt_freq(0.1)
        .crossover("sbx", prob=1.0, eta=30.0)
        .mutation("polynomial", prob=1.0 / 6.0, eta=20.0)
        .build()
    )
    algo = RVEA(cfg.to_dict(), kernel=NumPyKernel())
    recorder = RecorderViz()

    result = algo.run(DTLZ2Problem(n_var=6, n_obj=3), termination=("max_evaluations", 20), seed=3, live_viz=recorder)

    assert result["F"].shape[0] > 0
    assert recorder.starts == 1
    assert recorder.ends == 1
    assert recorder.gens >= 1


def _contains(limits, low, high):
    return limits[0] <= low and limits[1] >= high


def test_live_plot_rescales_axes_when_population_moves(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    viz = LiveParetoPlot(update_interval=1, interactive_backend="Agg")
    viz.on_start()
    rng = np.random.default_rng(0)
    first = np.column_stack([rng.random(20), 2.0 + 3.0 * rng.random(20)])
    f1 = np.linspace(0.0, 1.0, 20)
    converged = np.column_stack([f1, 1.0 - np.sqrt(f1)])  # ZDT1 front

    viz.on_generation(0, F=first)
    viz.on_generation(5, F=converged)

    assert _contains(viz.axes.get_ylim(), 0.0, 1.0)
    assert _contains(viz.axes.get_xlim(), 0.0, 1.0)


def test_live_plot_rescales_3d_axes(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    viz = LiveParetoPlot(update_interval=1, interactive_backend="Agg", objectives_to_plot=(0, 1, 2))
    viz.on_start()
    viz.on_generation(0, F=np.full((5, 3), 4.0) + np.arange(5)[:, None])
    viz.on_generation(1, F=np.eye(3))

    assert _contains(viz.axes.get_zlim(), 0.0, 1.0) and viz.axes.get_zlim()[1] < 4.0


def test_live_plot_labels_the_plotted_objectives(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    viz = LiveParetoPlot(interactive_backend="Agg", objectives_to_plot=(2, 0))
    viz.on_start()
    viz.on_generation(0, F=np.array([[0.1, 0.5, 0.9], [0.4, 0.2, 0.3]]))

    assert (viz.axes.get_xlabel(), viz.axes.get_ylabel()) == ("Objective 3", "Objective 1")
    assert "population" in viz.axes.get_title().lower()
    np.testing.assert_allclose(viz.scatter.get_xdata(), [0.9, 0.3])


def test_live_plot_subsample_uses_its_own_generator(monkeypatch):
    import random

    monkeypatch.setenv("MPLBACKEND", "Agg")
    F = np.arange(200, dtype=float).reshape(100, 2)
    state = random.getstate()

    picks = [LiveParetoPlot(max_points=10, rng_seed=3)._maybe_subsample(F) for _ in range(2)]

    assert random.getstate() == state
    np.testing.assert_array_equal(picks[0], picks[1])
    assert picks[0].shape == (10, 2)


def test_live_pareto_plot_saves_file(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    viz = LiveParetoPlot(save_final_path=tmp_path / "live.png", update_interval=1, max_points=10)
    viz.on_start()
    F = np.array([[0.1, 0.9], [0.2, 0.8], [0.5, 0.5]])
    viz.on_generation(0, F=F)
    viz.on_end(final_F=F)
    assert (tmp_path / "live.png").exists()
