import warnings

import numpy as np
import pytest

import vamos.engine.algorithm.components.subset_selection as ss_mod
import vamos.foundation.quality_indicators.hypervolume as hv_mod
import vamos.foundation.quality_indicators.moocore_indicators as mi_mod
from vamos.engine.algorithm.components.archive import (
    CrowdingDistanceArchive,
    HypervolumeArchive,
    MaxMinArchive,
    ReferenceDirectionsArchive,
    SPEA2Archive,
    UnboundedArchive,
)
from vamos.engine.algorithm.components.subset_selection import select_top_k_crowding, select_top_k_hv_greedy
from vamos.engine.algorithm.config import SMPSOConfig
from vamos.engine.algorithm.smpso import SMPSO
from vamos.engine.algorithm.spea2.helpers import truncate_by_distance
from vamos.engine.archive import ExternalArchiveConfig
from vamos.engine.archive.factory import setup_result_archive
from vamos.foundation.exceptions import DependencyError
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


def _tradeoff_front(n: int) -> np.ndarray:
    x = np.arange(n, dtype=float)
    y = (n - x).astype(float)
    return np.column_stack([x, y])


def test_hypervolume_archive_trims_by_contribution():
    archive = HypervolumeArchive(capacity=2, n_var=2, n_obj=2, dtype=float)
    X = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]])
    F = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    archive.update(X, F)
    _, result_F = archive.contents()
    assert result_F.shape[0] == 2
    # Middle point should be trimmed (lowest HV contribution).
    assert not np.any(np.all(result_F == np.array([0.5, 0.5]), axis=1))


def test_hysteresis_truncate_size_behavior():
    archive = CrowdingDistanceArchive(capacity=20, truncate_size=10, n_var=1, n_obj=2, dtype=float)

    X0 = np.arange(25, dtype=float).reshape(-1, 1)
    F0 = _tradeoff_front(25)
    archive.update(X0, F0)
    assert archive.contents()[1].shape[0] == 10

    X1 = np.arange(25, 30, dtype=float).reshape(-1, 1)
    F1_x = np.arange(25, 30, dtype=float)
    F1 = np.column_stack([F1_x, 25.0 - F1_x])
    archive.update(X1, F1)
    assert archive.contents()[1].shape[0] == 15

    X2 = np.arange(30, 40, dtype=float).reshape(-1, 1)
    F2_x = np.arange(30, 40, dtype=float)
    F2 = np.column_stack([F2_x, 25.0 - F2_x])
    archive.update(X2, F2)
    assert archive.contents()[1].shape[0] == 10


def test_spea2_archive_truncates_to_target_size():
    archive = SPEA2Archive(capacity=5, n_var=1, n_obj=3, dtype=float)
    X = np.arange(12, dtype=float).reshape(-1, 1)
    F = np.column_stack(
        [
            np.linspace(0.0, 1.0, 12),
            np.linspace(1.0, 0.0, 12),
            np.abs(np.linspace(-0.5, 0.5, 12)),
        ]
    )
    archive.update(X, F)
    _, kept_F = archive.contents()
    assert kept_F.shape[0] == 5


def _reference_truncate_by_distance(dist_matrix: np.ndarray, keep: int) -> np.ndarray:
    candidates = list(range(dist_matrix.shape[0]))
    if len(candidates) <= keep:
        return np.asarray(candidates, dtype=int)

    dist = np.asarray(dist_matrix, dtype=float)
    while len(candidates) > keep:
        sub = dist[np.ix_(candidates, candidates)]
        np.fill_diagonal(sub, np.inf)
        second_neighbor = np.partition(sub, 1, axis=1)[:, 1]
        remove_pos = int(np.argmin(second_neighbor))
        del candidates[remove_pos]

    return np.asarray(candidates, dtype=int)


def test_truncate_by_distance_matches_reference_selection_rule():
    rng = np.random.default_rng(0)
    for n, keep in [(8, 4), (11, 5), (14, 6)]:
        F = rng.random((n, 3))
        dist = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
        expected = _reference_truncate_by_distance(dist, keep)
        actual = truncate_by_distance(dist, keep)
        np.testing.assert_array_equal(actual, expected)


def test_spea2_archive_skips_strength_ranking_for_nondominated_front(monkeypatch: pytest.MonkeyPatch):
    import vamos.engine.algorithm.spea2.helpers as spea2_helpers

    def _unexpected_call(*_args, **_kwargs):
        raise AssertionError("dominance_matrix should not be used for archive truncation.")

    monkeypatch.setattr(spea2_helpers, "dominance_matrix", _unexpected_call)

    archive = SPEA2Archive(capacity=5, n_var=1, n_obj=2, dtype=float)
    X = np.arange(12, dtype=float).reshape(-1, 1)
    x = np.linspace(0.0, 1.0, 12)
    F = np.column_stack([x, 1.0 - x])

    archive.update(X, F)
    _, kept_F = archive.contents()
    assert kept_F.shape[0] == 5


def test_maxmin_archive_truncates_to_target_size():
    archive = MaxMinArchive(capacity=3, n_var=1, n_obj=2, dtype=float)
    X = np.arange(5, dtype=float).reshape(-1, 1)
    F = np.array(
        [
            [0.0, 10.0],
            [1.0, 9.0],
            [5.0, 5.0],
            [9.0, 1.0],
            [10.0, 0.0],
        ]
    )
    archive.update(X, F)
    _, kept_F = archive.contents()
    assert kept_F.shape[0] == 3
    assert np.any(np.all(kept_F == np.array([0.0, 10.0]), axis=1))
    assert np.any(np.all(kept_F == np.array([10.0, 0.0]), axis=1))


def test_ref_dirs_archive_truncates_to_target_size():
    archive = ReferenceDirectionsArchive(capacity=3, n_var=1, n_obj=3, dtype=float, rng_seed=0)
    X = np.arange(5, dtype=float).reshape(-1, 1)
    F = np.array(
        [
            [0.90, 0.05, 0.05],
            [0.05, 0.90, 0.05],
            [0.05, 0.05, 0.90],
            [0.34, 0.34, 0.34],
            [0.60, 0.20, 0.20],
        ]
    )
    archive.update(X, F)
    _, kept_F = archive.contents()
    assert kept_F.shape[0] == 3
    assert np.any(np.all(kept_F == np.array([0.90, 0.05, 0.05]), axis=1))
    assert np.any(np.all(kept_F == np.array([0.05, 0.90, 0.05]), axis=1))
    assert np.any(np.all(kept_F == np.array([0.05, 0.05, 0.90]), axis=1))


def test_archive_feasibility_prefers_feasible_points():
    archive = CrowdingDistanceArchive(capacity=10, n_var=1, n_obj=2, dtype=float, n_con=1)
    X = np.array([[0.0], [1.0], [2.0]])
    F = np.array([[0.1, 0.1], [1.0, 1.0], [2.0, 2.0]])
    G = np.array([[1.0], [0.0], [0.0]])

    archive.update(X, F, G)
    _, kept_F = archive.contents()
    assert kept_F.shape[0] == 1
    np.testing.assert_allclose(kept_F[0], np.array([1.0, 1.0]))


def test_archive_feasibility_keeps_least_violating_when_no_feasible():
    archive = CrowdingDistanceArchive(capacity=10, n_var=1, n_obj=2, dtype=float, n_con=1)
    X = np.array([[0.0], [1.0], [2.0]])
    F = np.array([[0.0, 0.0], [1.0, 0.5], [0.5, 1.0]])
    G = np.array([[2.0], [1.0], [1.0]])

    archive.update(X, F, G)
    _, kept_F = archive.contents()
    assert kept_F.shape[0] == 2
    assert not np.any(np.all(kept_F == np.array([0.0, 0.0]), axis=1))


def test_hypervolume_reference_is_stable():
    archive = HypervolumeArchive(capacity=2, n_var=1, n_obj=2, dtype=float, ref_offset=1.0)
    X0 = np.array([[0.0], [1.0], [2.0]])
    F0 = np.array([[10.0, 1.0], [1.0, 10.0], [6.0, 6.0]])
    archive.update(X0, F0)
    assert archive._global_worst is not None  # internal monotone reference tracker
    first_ref = archive._global_worst.copy()

    X1 = np.array([[3.0]])
    F1 = np.array([[0.1, 0.1]])
    archive.update(X1, F1)
    assert archive._global_worst is not None
    assert np.all(archive._global_worst >= first_ref)


def test_decision_space_dedup_keeps_equal_objectives_with_distinct_decisions():
    archive = UnboundedArchive(
        n_var=2,
        n_obj=2,
        dtype=float,
        deduplicate_in="decision",
        decision_tolerance=1e-32,
    )
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    F = np.array([[1.0, 1.0], [1.0 + 1e-12, 1.0 - 1e-12]])
    archive.update(X, F)
    _, kept_F = archive.contents()
    assert kept_F.shape[0] == 2


def test_objective_space_dedup_collapses_equal_objectives():
    archive = UnboundedArchive(
        n_var=2,
        n_obj=2,
        dtype=float,
        deduplicate_in="objective",
        objective_tolerance=1e-10,
    )
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    F = np.array([[1.0, 1.0], [1.0 + 1e-12, 1.0 - 1e-12]])
    archive.update(X, F)
    _, kept_F = archive.contents()
    assert kept_F.shape[0] == 1


@pytest.fixture
def without_moocore(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the hypervolume helpers as in an installation without moocore."""
    monkeypatch.setattr(hv_mod, "_OPTIONAL_LOADED", True)
    monkeypatch.setattr(hv_mod, "_MOOCORE", None)
    monkeypatch.setattr(hv_mod, "_LIBHV_MODULE", None)
    monkeypatch.setattr(mi_mod, "_moocore", None)
    monkeypatch.setattr(ss_mod, "_moocore", None)


def _permuted_zdt1_front() -> np.ndarray:
    f1 = np.linspace(0.0, 1.0, 21)
    front = np.column_stack([f1, 1.0 - np.sqrt(f1)])
    return front[np.random.default_rng(0).permutation(21)]


def _sphere_front(n: int, n_obj: int, seed: int) -> np.ndarray:
    points = np.random.default_rng(seed).random((n, n_obj)) + 0.05
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def _brute_force_contributions(F: np.ndarray, ref: np.ndarray) -> np.ndarray:
    full = hv_mod._hypervolume_impl(F, ref)
    return np.array([full - hv_mod._hypervolume_impl(np.delete(F, i, axis=0), ref) for i in range(F.shape[0])])


def test_hv_contributions_without_moocore_are_exact(without_moocore: None) -> None:
    F2 = _permuted_zdt1_front()
    ref2 = np.array([1.1, 1.1])
    F3 = _sphere_front(12, 3, seed=1)
    ref3 = np.full(3, 1.1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        contrib2 = ss_mod._hv_contributions(F2, ref2)
        contrib3 = ss_mod._hv_contributions(F3, ref3)
        selected = select_top_k_hv_greedy(F2, 5, ref_point=ref2)

    np.testing.assert_allclose(contrib2, _brute_force_contributions(F2, ref2), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(contrib3, _brute_force_contributions(F3, ref3), rtol=1e-12, atol=1e-15)
    # moocore selects these indices on the same front (HV 0.7663).
    assert selected.tolist() == [14, 4, 17, 0, 1]
    assert hv_mod._hypervolume_impl(F2[selected], ref2) == pytest.approx(0.7663, abs=1e-4)


@pytest.mark.parametrize("n_obj", [2, 3, 4])
def test_hv_contributions_without_moocore_match_moocore(without_moocore: None, n_obj: int) -> None:
    moocore = pytest.importorskip("moocore")
    rng = np.random.default_rng(10 + n_obj)
    front = _sphere_front(6, n_obj, seed=n_obj)
    dominated = front[:2] + 0.05
    outside = front[2:3] + np.eye(n_obj)[0] * 2.0
    on_boundary = front[3:4].copy()
    on_boundary[0, -1] = 1.2
    F = np.vstack([front, dominated, outside, on_boundary, front[4:5]])
    F = F[rng.permutation(F.shape[0])]
    ref = np.full(n_obj, 1.2)

    expected = np.asarray(moocore.hv_contributions(F, ref=ref), dtype=float)

    np.testing.assert_allclose(ss_mod._hv_contributions(F, ref), expected, rtol=1e-12, atol=1e-15)


def test_hv_contributions_delegate_to_moocore_without_python_front_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    moocore = pytest.importorskip("moocore")

    def _python_front_filter(*args: object, **kwargs: object) -> None:
        raise AssertionError("the pure-Python front filter must not run when moocore is installed")

    monkeypatch.setattr(ss_mod, "pareto_filter", _python_front_filter)
    front = _sphere_front(12, 3, seed=2)
    F = np.vstack([front, front[:2] + 0.05, front[2:3] + np.array([2.0, 0.0, 0.0])])
    ref = np.full(3, 1.2)

    expected = np.asarray(moocore.hv_contributions(F, ref=ref), dtype=float)
    np.testing.assert_allclose(ss_mod._hv_contributions(F, ref), expected, rtol=1e-12, atol=1e-15)

    archive = HypervolumeArchive(capacity=5, n_var=1, n_obj=3, dtype=float, ref_point=ref)
    _, kept = archive.update(np.arange(front.shape[0], dtype=float)[:, None], front)
    assert kept.shape[0] == 5


def test_hypervolume_archive_prunes_exactly_without_moocore(without_moocore: None) -> None:
    F = _permuted_zdt1_front()
    ref = np.array([1.1, 1.1])
    archive = HypervolumeArchive(capacity=5, n_var=1, n_obj=2, dtype=float, ref_point=ref)

    _, kept = archive.update(np.arange(F.shape[0], dtype=float)[:, None], F)

    keep = list(range(F.shape[0]))
    while len(keep) > 5:
        del keep[int(np.argmin(_brute_force_contributions(F[keep], ref)))]
    expected = F[keep]
    assert sorted(map(tuple, kept)) == sorted(map(tuple, expected))


def test_hypervolume_archive_requires_moocore_above_two_objectives(without_moocore: None) -> None:
    with pytest.raises(DependencyError, match="'moocore' is required for hypervolume archive pruning with 3 objectives"):
        HypervolumeArchive(capacity=5, n_var=1, n_obj=3, dtype=float)
    with pytest.raises(DependencyError, match="moocore"):
        setup_result_archive(ExternalArchiveConfig(capacity=5, pruning="hv"), n_var=0, n_obj=3, dtype=float)

    assert isinstance(HypervolumeArchive(capacity=5, n_var=1, n_obj=2, dtype=float), HypervolumeArchive)


def test_select_top_k_crowding_returns_k_indices():
    F = _tradeoff_front(20)
    idx = select_top_k_crowding(F, 5)
    assert idx.shape == (5,)
    assert len(set(idx.tolist())) == 5
    assert all(0 <= i < 20 for i in idx)


def test_select_top_k_crowding_returns_all_when_k_ge_n():
    F = _tradeoff_front(3)
    idx = select_top_k_crowding(F, 10)
    assert idx.shape == (3,)


def test_select_top_k_crowding_keeps_extremes():
    F = np.array([[0.0, 10.0], [5.0, 5.0], [10.0, 0.0], [4.0, 5.5]])
    idx = select_top_k_crowding(F, 2)
    assert 0 in idx and 2 in idx


def test_select_top_k_crowding_rejects_non_positive_k():
    F = _tradeoff_front(5)
    with pytest.raises(ValueError, match="k must be a positive"):
        select_top_k_crowding(F, 0)


def test_nsgaii_constrained_with_external_archive():
    """Constrained NSGA-II with external archive must not crash on G shape mismatch.

    Regression test: ``tell_nsgaii`` used to pass ``combined_X/combined_F``
    (pop_size + offspring_size rows) to ``update_archives`` while ``G`` defaulted
    to ``state.G`` (pop_size rows after survival), causing a shape mismatch in the
    archive's feasibility filter.
    """
    from vamos.engine.algorithm.config import NSGAIIConfig
    from vamos.engine.algorithm.nsgaii import NSGAII
    from vamos.foundation.kernel.numpy_backend import NumPyKernel

    class _ConstrainedBiobj:
        n_var = 2
        n_obj = 2
        n_constraints = 1
        xl = np.array([0.0, 0.0])
        xu = np.array([1.0, 1.0])
        encoding = "real"

        def evaluate(self, X, out):
            out["F"] = np.column_stack([X[:, 0], 1.0 - X[:, 0] + X[:, 1]])
            out["G"] = (X[:, 0] + X[:, 1] - 1.5)[:, None]

    pop = 12
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(pop)
        .offspring_size(pop)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .constraint_mode("feasibility")
        .external_archive(capacity=pop * 2, pruning="crowding")
        .build()
    )

    algo = NSGAII(cfg.to_dict(), kernel=NumPyKernel())
    problem = _ConstrainedBiobj()
    # Must complete without ValueError from shape mismatch
    result = algo.run(problem, termination=("max_evaluations", pop * 4), seed=42)

    assert result["F"].shape[0] > 0
    assert "archive" in result
    assert result["archive"]["F"].shape[0] > 0


@pytest.mark.parametrize("pruning", ["maxmin", "ref_dirs"])
def test_nsgaii_runs_with_new_external_archive_policies(pruning: str):
    from vamos.engine.algorithm.config import NSGAIIConfig
    from vamos.engine.algorithm.nsgaii import NSGAII
    from vamos.foundation.kernel.numpy_backend import NumPyKernel

    class _ThreeObj:
        n_var = 3
        n_obj = 3
        n_constraints = 0
        xl = np.zeros(3)
        xu = np.ones(3)
        encoding = "real"

        def evaluate(self, X, out):
            out["F"] = np.column_stack(
                [
                    X[:, 0],
                    1.0 - X[:, 0] + 0.2 * X[:, 1],
                    0.5 * X[:, 1] + 1.0 - X[:, 2],
                ]
            )

    pop = 12
    cfg = (
        NSGAIIConfig.builder()
        .pop_size(pop)
        .offspring_size(pop)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .external_archive(capacity=pop * 2, pruning=pruning)
        .build()
    )

    algo = NSGAII(cfg.to_dict(), kernel=NumPyKernel())
    result = algo.run(_ThreeObj(), termination=("max_evaluations", pop * 4), seed=7)

    assert result["F"].shape[0] > 0
    assert "archive" in result
    assert result["archive"]["F"].shape[0] > 0


class _ScaledZDT1(ZDT1Problem):
    def __init__(self, n_var: int, scale: float) -> None:
        super().__init__(n_var)
        self.scale = float(scale)

    def evaluate(self, X: np.ndarray, out: dict[str, np.ndarray]) -> None:
        super().evaluate(X, out)
        out["F"] *= self.scale


def _zdt1_front_points(n: int) -> tuple[np.ndarray, np.ndarray]:
    f1 = np.linspace(0.0, 1.0, n)
    F = np.column_stack([f1, 1.0 - np.sqrt(f1)])
    X = np.column_stack([f1, np.zeros(n), np.zeros(n)])
    return X, F


@pytest.mark.parametrize("scale", [1.0, 1e-8, 1e-10, 1e-12])
def test_crowding_archive_keeps_distinct_points_at_tiny_scales(scale):
    X, F = _zdt1_front_points(10)
    archive = CrowdingDistanceArchive(20, 3, 2, float)
    archive.update(X, F * scale)
    _, kept_F = archive.contents()
    assert kept_F.shape[0] == 10, f"archive keeps {kept_F.shape[0]} of 10 distinct points at scale {scale}"


def test_archives_deduplicate_identical_objectives_exactly():
    X, F = _zdt1_front_points(5)
    X_dup = np.vstack([X, X + 0.5])
    F_dup = np.vstack([F, F])
    for archive in (
        CrowdingDistanceArchive(20, 3, 2, float),
        UnboundedArchive(n_var=3, n_obj=2, dtype=float),
        HypervolumeArchive(20, 3, 2, float),
        SPEA2Archive(20, 3, 2, float),
        MaxMinArchive(20, 3, 2, float),
    ):
        archive.update(X_dup, F_dup)
        assert archive.contents()[1].shape[0] == 5, type(archive).__name__
    assert ExternalArchiveConfig().objective_tolerance == 0.0


def test_smpso_leaders_do_not_depend_on_the_objective_scale():
    cfg = SMPSOConfig.builder().pop_size(20).archive_size(20).mutation("polynomial", prob=0.1, eta=20.0).build().to_dict()

    def run(scale: float) -> dict:
        problem = _ScaledZDT1(n_var=6, scale=scale)
        return SMPSO(cfg, kernel=NumPyKernel()).run(problem, termination=("max_evaluations", 600), seed=3)

    reference = run(1.0)
    scaled = run(1e-11)
    assert scaled["X"].shape == reference["X"].shape
    assert np.array_equal(scaled["X"], reference["X"])
