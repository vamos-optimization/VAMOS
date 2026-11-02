import copy
from typing import Any

import numpy as np
import pytest

from vamos.engine.algorithm.components.archive import CrowdingDistanceArchive
from vamos.engine.algorithm.config import SMPSOConfig
from vamos.engine.algorithm.smpso import SMPSO
from vamos.engine.algorithm.smpso.helpers import update_personal_bests
from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


def test_smpso_numba_same_seed_reproducible():
    cfg = (SMPSOConfig.builder().pop_size(30).archive_size(30).mutation("polynomial", prob=0.1, eta=20.0).build()).to_dict()
    problem = ZDT1Problem(n_var=8)

    result_a = SMPSO(cfg, kernel=NumbaKernel()).run(problem, termination=("max_evaluations", 90), seed=17)
    result_b = SMPSO(cfg, kernel=NumbaKernel()).run(problem, termination=("max_evaluations", 90), seed=17)

    assert np.array_equal(result_a["F"], result_b["F"])
    assert np.array_equal(result_a["X"], result_b["X"])


class _RecordingRng:
    """A generator proxy that records the arrays drawn by ``uniform``."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self.uniform_draws: list[np.ndarray] = []

    def uniform(self, *args: object, **kwargs: object) -> np.ndarray:
        value = np.asarray(self._rng.uniform(*args, **kwargs))
        self.uniform_draws.append(value.copy())
        return value

    def __getattr__(self, name: str) -> object:
        return getattr(self._rng, name)


def _initialized_smpso(pop_size: int = 8, archive_size: int = 8, seed: int = 4) -> SMPSO:
    cfg = SMPSOConfig.builder().pop_size(pop_size).archive_size(archive_size).mutation("polynomial", prob=0.25, eta=20.0).build()
    algorithm = SMPSO(cfg.to_dict(), kernel=NumPyKernel())
    algorithm.initialize(ZDT1Problem(n_var=4), ("max_evaluations", 200), seed=seed)
    return algorithm


def test_smpso_coefficients_are_not_rounded() -> None:
    algorithm = _initialized_smpso()
    st = algorithm._st
    assert st is not None
    recorder = _RecordingRng(st.rng)
    st.rng = recorder
    # Zero velocity, pbest = X and a single leader leave constriction * c2 * r2 * (leader - X).
    st.velocity[:] = 0.0
    st.pbest_X[:] = st.X
    leader = st.X[:1].copy()
    st.archive_X, st.archive_F, st.archive_crowding = leader, st.F[:1].copy(), None
    X = st.X.copy()

    algorithm.ask()

    r1, r2, c1, c2 = recorder.uniform_draws[:4]
    assert np.any(np.abs(c2 * 10.0 - np.round(c2 * 10.0)) > 1e-9)
    rho = c1 + c2
    constriction = np.where(rho <= 4.0, 1.0, 2.0 / (2.0 - rho - np.sqrt(np.maximum(rho * rho - 4.0 * rho, 0.0))))
    expected = np.clip(constriction * c2 * r2 * (leader - X), st.delta_min, st.delta_max)
    unchanged = (X + expected >= st.xl) & (X + expected <= st.xu)
    np.testing.assert_allclose(st.velocity[unchanged], expected[unchanged], rtol=1e-12, atol=0.0)


@pytest.mark.parametrize(
    ("G", "pbest_G", "mode"),
    [
        (None, None, "none"),
        (np.array([[-1.0]]), np.array([[-0.5]]), "feasibility"),
        (np.array([[0.5]]), np.array([[0.5]]), "feasibility"),
    ],
    ids=["unconstrained", "both-feasible", "equal-violation"],
)
def test_smpso_pbest_rule_is_the_same_with_and_without_constraints(G, pbest_G, mode) -> None:  # noqa: ANN001
    for new_F, expected in ((np.array([[1.0, 2.0]]), 0.2), (np.array([[3.0, 2.0]]), 0.9)):
        pbest_X = np.array([[0.9]])
        pbest_F = np.array([[2.0, 1.0]])
        update_personal_bests(
            np.array([[0.2]]), new_F, None if G is None else G.copy(), pbest_X, pbest_F, None if pbest_G is None else pbest_G.copy(), mode
        )
        assert pbest_X[0, 0] == expected


def _sequential(archive: Any, X: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    contents = archive.contents()
    for row in range(X.shape[0]):
        contents = archive.update(X[row : row + 1], F[row : row + 1])
    return contents


def test_leaders_archive_matches_sequential_insertion() -> None:
    problem = ZDT1Problem(n_var=4)
    algorithm = _initialized_smpso(pop_size=30, archive_size=4, seed=2)
    st = algorithm._st
    assert st is not None
    empty = CrowdingDistanceArchive(4, 4, 2, float)
    initial_X, initial_F = _sequential(empty, st.X, st.F)
    assert st.archive_F.tobytes() == initial_F.tobytes()
    assert st.archive_X.tobytes() == initial_X.tobytes()

    for _ in range(3):
        reference = copy.deepcopy(st.archive_manager)
        X_new = algorithm.ask()
        out = {"F": np.empty((X_new.shape[0], 2))}
        problem.evaluate(X_new, out)
        expected_X, expected_F = _sequential(reference, X_new, out["F"])
        algorithm.tell(out)
        assert st.archive_F.tobytes() == expected_F.tobytes()
        assert st.archive_X.tobytes() == expected_X.tobytes()
