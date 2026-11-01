"""NumPy and Numba engines must produce identical trajectories for built-in algorithms."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

import vamos
from vamos.engine.hooks.live_viz import NoOpLiveVisualization
from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.foundation.kernel.numpy_backend import NumPyKernel

ALGORITHMS = ("nsgaii", "nsgaiii", "moead", "smsemoa", "spea2", "ibea", "smpso", "agemoea", "rvea")
PROBLEMS = (
    ("zdt1", {}),
    ("c1dtlz1", {"n_obj": 3}),
)


def _final_arrays(problem: str, algorithm: str, engine: str, kwargs: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    result = vamos.optimize(problem, algorithm=algorithm, engine=engine, seed=3, max_evaluations=600, **kwargs)
    return np.asarray(result.X), np.asarray(result.F)


@pytest.mark.numba
@pytest.mark.parametrize("problem,kwargs", PROBLEMS, ids=[name for name, _ in PROBLEMS])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_numpy_and_numba_engines_follow_the_same_trajectory(algorithm: str, problem: str, kwargs: dict[str, int]) -> None:
    X_np, F_np = _final_arrays(problem, algorithm, "numpy", kwargs)
    X_nb, F_nb = _final_arrays(problem, algorithm, "numba", kwargs)

    np.testing.assert_array_equal(X_nb, X_np)
    np.testing.assert_array_equal(F_nb, F_np)


@pytest.mark.numba
@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("problem,kwargs", PROBLEMS, ids=[name for name, _ in PROBLEMS])
def test_auto_preserves_generation_trace_and_generator_states(algorithm, problem, kwargs, monkeypatch):
    """Compare intermediate populations and every run-owned RNG, not just fronts."""
    from copy import deepcopy

    from vamos.experiment.runtime.catalog import resolve_engine

    assert resolve_engine("auto", algorithm=algorithm) == "numba"
    generators = []
    original = np.random.default_rng

    def tracked_rng(*args, **kwargs):
        rng = original(*args, **kwargs)
        generators.append(rng)
        return rng

    class Trace(NoOpLiveVisualization):
        def __init__(self):
            self.frames = []

        def on_generation(self, generation, F=None, X=None, stats=None):
            self.frames.append((generation, None if F is None else F.copy(), None if X is None else X.copy()))

    monkeypatch.setattr(np.random, "default_rng", tracked_rng)
    runs = []
    for engine in ("numpy", "numba", "auto"):
        generators.clear()
        trace = Trace()
        result = vamos.optimize(problem, algorithm=algorithm, engine=engine, seed=17, max_evaluations=600, live_viz=trace, **kwargs)
        states = [deepcopy(rng.bit_generator.state) for rng in generators]
        assert states and trace.frames
        assert result.data["evaluations"] == 600
        runs.append((result, trace.frames, states))
    reference, frames, states = runs[0]
    for result, actual_frames, actual_states in runs[1:]:
        assert actual_states == states
        assert len(actual_frames) == len(frames)
        for expected, actual in zip(frames, actual_frames):
            assert expected[0] == actual[0]
            for left, right in zip(expected[1:], actual[1:]):
                if left is None:
                    assert right is None
                else:
                    assert left.dtype == right.dtype and left.shape == right.shape
                    assert left.tobytes() == right.tobytes()
        for name in ("X", "F"):
            assert getattr(reference, name).tobytes() == getattr(result, name).tobytes()
    assert runs[-1][0].meta["engine"] == "numba"
    assert runs[-1][0].meta["default_sources"]["engine"] == "auto"


@pytest.mark.numba
def test_compiled_variation_runs_only_for_the_numba_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    kernels = NumbaKernel().variation_kernels()
    assert NumPyKernel().variation_kernels() is None
    if kernels is None:
        pytest.skip("compiled pow differs from np.power on this platform; compiled variation stays disabled")
    calls = {"sbx": 0, "pm": 0}
    original_sbx = kernels.sbx_crossover_pairs
    original_pm = kernels.polynomial_mutation

    def counting_sbx(*args, **kwargs):
        calls["sbx"] += 1
        return original_sbx(*args, **kwargs)

    def counting_pm(*args, **kwargs):
        calls["pm"] += 1
        return original_pm(*args, **kwargs)

    monkeypatch.setattr(kernels, "sbx_crossover_pairs", counting_sbx)
    monkeypatch.setattr(kernels, "polynomial_mutation", counting_pm)

    # Generational NSGA-II mates many pairs per step, so no single-pair shortcut applies.
    vamos.optimize("zdt1", algorithm="nsgaii", engine="numpy", seed=1, pop_size=20, max_evaluations=100)
    assert calls == {"sbx": 0, "pm": 0}

    vamos.optimize("zdt1", algorithm="nsgaii", engine="numba", seed=1, pop_size=20, max_evaluations=100)
    assert calls["sbx"] > 0
    assert calls["pm"] > 0


@pytest.mark.parametrize("problem,kwargs", PROBLEMS, ids=[name for name, _ in PROBLEMS])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_numpy_and_moocore_engines_follow_the_same_trajectory(algorithm: str, problem: str, kwargs: dict[str, int]) -> None:
    pytest.importorskip("moocore")
    X_np, F_np = _final_arrays(problem, algorithm, "numpy", kwargs)
    X_mc, F_mc = _final_arrays(problem, algorithm, "moocore", kwargs)

    np.testing.assert_array_equal(X_mc, X_np)
    np.testing.assert_array_equal(F_mc, F_np)
