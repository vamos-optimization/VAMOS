import numpy as np
import pytest

from vamos.engine.algorithm.config import SPEA2Config
from vamos.engine.algorithm.spea2 import SPEA2
from vamos.engine.algorithm.spea2.helpers import (
    dominance_matrix,
    environmental_selection,
    knn_density,
    spea2_fitness,
    strength_raw_fitness,
)
from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


def _selection_kernels(engine: str):
    if engine == "numba":
        pytest.importorskip("numba")
    return resolve_kernel(engine).selection_kernels()


@pytest.mark.parametrize("engine", ["numpy", "numba"])
@pytest.mark.parametrize("reverse_rows", [False, True])
def test_spea2_truncation_breaks_ties_lexicographically(engine: str, reverse_rows: bool) -> None:
    # (0, 1) and (0.1, 0.9) share the smallest nearest-neighbour distance; the
    # SPEA2 truncation operator (Zitzler et al. 2001; jMetal) compares the second
    # nearest distance next and always removes (0.1, 0.9), never the extreme.
    F = np.array([[0.0, 1.0], [0.1, 0.9], [0.5, 0.5], [1.0, 0.0]])
    if reverse_rows:
        F = F[::-1].copy()
    X = np.arange(4, dtype=float)[:, None]

    _, kept_F, *_ = environmental_selection(X, F, None, 3, 1, "none", _selection_kernels(engine))

    assert {tuple(row) for row in kept_F.tolist()} == {(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)}


def _reference_lexicographic_truncation(dist: np.ndarray, keep: int) -> list[int]:
    """Remove the row whose sorted distances to the others are lexicographically smallest."""
    active = list(range(dist.shape[0]))
    while len(active) > keep:
        keys = [tuple(sorted(dist[i, j] for j in active if j != i)) for i in active]
        active.pop(keys.index(min(keys)))
    return active


@pytest.mark.parametrize("engine", ["numpy", "numba"])
def test_spea2_truncation_matches_the_reference_operator(engine: str) -> None:
    rng = np.random.default_rng(5)
    for n_obj, n in [(2, 9), (3, 12), (4, 15)]:
        W = rng.random((n, n_obj))
        F = W / W.sum(axis=1, keepdims=True)
        dist = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
        _, kept_F, *_ = environmental_selection(np.arange(n, dtype=float)[:, None], F, None, 5, 1, "none", _selection_kernels(engine))
        np.testing.assert_array_equal(kept_F, F[_reference_lexicographic_truncation(dist, 5)])


def test_spea2_fills_with_dominated_by_fitness() -> None:
    # (0, 1) and (1, 0) are non-dominated; the two others are dominated only by
    # (0, 1), so both have raw fitness 2. SPEA2 fills the archive with the
    # dominated solution of lower F = R + D, where D = 1 / (sigma_1 + 2) uses the
    # nearest neighbour in the whole union: (0.02, 3) is far from every point,
    # while (0.05, 1.05) crowds (0, 1).
    F = np.array([[0.0, 1.0], [1.0, 0.0], [0.05, 1.05], [0.02, 3.0]])
    X = np.arange(4, dtype=float)[:, None]

    _, kept_F, *_ = environmental_selection(X, F, None, 3, 1, "none")

    assert {tuple(row) for row in kept_F.tolist()} == {(0.0, 1.0), (1.0, 0.0), (0.02, 3.0)}


class _IdentityProblem:
    n_var = 2
    n_obj = 2
    xl = np.zeros(2)
    xu = np.ones(2)
    encoding = "real"

    def evaluate(self, X, out):
        out["F"] = np.asarray(X, dtype=float).copy()


def test_spea2_tournament_uses_union_fitness() -> None:
    # SPEA2 assigns fitness to the union of archive and offspring and uses it for
    # mating selection on the new archive (Zitzler et al. 2001, steps 2 and 5).
    cfg = SPEA2Config.default(pop_size=8, n_var=2).to_dict()
    algorithm = SPEA2(cfg)
    algorithm.initialize(_IdentityProblem(), ("max_evaluations", 40), seed=1)
    state = algorithm._st
    assert state is not None
    for _ in range(3):
        archive_F = state.env_F.copy()
        offspring_F = algorithm.ask().copy()
        algorithm.tell(offspring_F)

        union = np.vstack([archive_F, offspring_F])
        dom, _, _ = dominance_matrix(union, None, "none")
        rows = [int(np.flatnonzero(np.all(union == row, axis=1))[0]) for row in state.env_F]
        np.testing.assert_array_equal(state.selection_raw_fitness, strength_raw_fitness(dom)[rows])
        np.testing.assert_array_equal(state.selection_density, knn_density(union, 1)[rows])


def test_spea2_numba_same_seed_reproducible():
    cfg = (
        SPEA2Config.builder()
        .pop_size(20)
        .archive_size(20)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .build()
    ).to_dict()
    problem = ZDT1Problem(n_var=8)

    result_a = SPEA2(cfg, kernel=NumbaKernel()).run(problem, termination=("max_evaluations", 60), seed=11)
    result_b = SPEA2(cfg, kernel=NumbaKernel()).run(problem, termination=("max_evaluations", 60), seed=11)

    assert np.array_equal(result_a["F"], result_b["F"])
    assert np.array_equal(result_a["X"], result_b["X"])


def _spea2_fitness_reference(F: np.ndarray, dom: np.ndarray, k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    n = F.shape[0]
    if n == 0:
        return np.empty(0), np.empty((0, 0))
    if k is None:
        k = max(1, int(np.sqrt(n)))
    k = min(k, n - 1) if n > 1 else 1

    strength = dom.sum(axis=1)
    raw_fitness = np.zeros(n, dtype=float)
    for i in range(n):
        dominators = np.where(dom[:, i])[0]
        raw_fitness[i] = strength[dominators].sum()

    dist = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            delta = np.linalg.norm(F[i] - F[j])
            dist[i, j] = delta
            dist[j, i] = delta

    if n == 1:
        density = np.array([0.0], dtype=float)
    else:
        density = np.zeros(n, dtype=float)
        for i in range(n):
            sorted_dists = np.sort(dist[i])
            sigma_k = sorted_dists[k] if k < n else sorted_dists[-1]
            density[i] = 1.0 / (sigma_k + 2.0)
    return raw_fitness + density, dist


def test_spea2_fitness_matches_reference_loop_implementation() -> None:
    F = np.array(
        [
            [0.1, 0.9, 0.5],
            [0.4, 0.6, 0.4],
            [0.7, 0.3, 0.6],
            [0.2, 0.8, 0.7],
        ],
        dtype=float,
    )
    dom, _, _ = dominance_matrix(F, None, "none")

    expected_fitness, expected_dist = _spea2_fitness_reference(F, dom, k=2)
    actual_fitness, actual_dist = spea2_fitness(F, dom, k=2)

    np.testing.assert_allclose(actual_fitness, expected_fitness)
    np.testing.assert_allclose(actual_dist, expected_dist)
