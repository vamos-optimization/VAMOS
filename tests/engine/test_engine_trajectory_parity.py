"""NumPy and Numba engines must produce identical trajectories for built-in algorithms."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

import vamos
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
