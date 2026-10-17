"""NumPy and Numba engines must produce identical trajectories for built-in algorithms."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

import vamos

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
