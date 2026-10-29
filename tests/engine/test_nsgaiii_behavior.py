from itertools import permutations
from math import comb

import numpy as np
import pytest

from vamos import optimize
from vamos.engine.algorithm.components.weight_vectors import _generate_weights
from vamos.engine.algorithm.config import NSGAIIIConfig
from vamos.engine.algorithm.factory import build_algorithm
from vamos.engine.algorithm.nsgaiii import NSGAIII, associate, nsgaiii_survival
from vamos.foundation.core.experiment_config import ExperimentConfig
from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.dtlz import DTLZ2Problem
from vamos.foundation.problem.zdt1 import ZDT1Problem


def _make_config(pop_size=10, divisions=3, prob="1/n"):
    return (
        NSGAIIIConfig.builder()
        .pop_size(pop_size)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob=prob, eta=20.0)
        .selection("tournament", size=2)
        .reference_directions(divisions=divisions)
        .build()
    ).to_dict()


def test_nsgaiii_survival_preserves_population_size_with_odd_pop():
    cfg = _make_config(pop_size=15, divisions=4)
    alg = NSGAIII(cfg, kernel=NumPyKernel())
    problem = DTLZ2Problem(n_var=12, n_obj=3)
    result = alg.run(problem, termination=("max_evaluations", 30), seed=7)
    assert result["X"].shape[0] == 15
    assert result["F"].shape[0] == 15


def test_reference_directions_truncate_when_excess():
    cfg = (
        NSGAIIIConfig.builder()
        .pop_size(6)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("polynomial", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .reference_directions(divisions=10)  # generates more than pop_size
        .build()
    ).to_dict()
    alg = NSGAIII(cfg, kernel=NumPyKernel())
    problem = ZDT1Problem(n_var=6)
    with pytest.raises(ValueError, match="pop_size"):
        alg.run(problem, termination=("max_evaluations", 12), seed=3)


def test_association_handles_degenerate_front():
    # Force a degenerate front: all identical objective vectors
    F = np.full((6, 2), 1.0)
    ref_dirs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    ref_dirs_norm = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    associations, distances = associate(F, ref_dirs_norm)
    assert associations.shape[0] == F.shape[0]
    assert np.isfinite(distances).all()


def test_directional_diversity_preserved():
    cfg = _make_config(pop_size=9)
    # Crafted objective values already aligned to distinct ref directions
    F = np.array(
        [
            [0.1, 0.9, 0.9],
            [0.9, 0.1, 0.9],
            [0.9, 0.9, 0.1],
            [0.2, 0.2, 0.8],
            [0.2, 0.8, 0.2],
            [0.8, 0.2, 0.2],
            [0.4, 0.4, 0.6],
            [0.4, 0.6, 0.4],
            [0.6, 0.4, 0.4],
        ]
    )
    X = np.zeros_like(F)
    rng = np.random.default_rng(0)
    n_obj = F.shape[1]
    ref_dirs = np.eye(n_obj)
    ref_dirs_norm = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    # Use public helper function instead of method
    # Simulate with empty offspring arrays for survival test
    ideal = np.full(n_obj, np.inf)
    worst = np.full(n_obj, -np.inf)
    X_sel, F_sel, _, _, _, _, _ = nsgaiii_survival(
        X,
        F,
        None,
        np.empty((0, X.shape[1])),
        np.empty((0, F.shape[1])),
        None,
        cfg["pop_size"],
        ref_dirs_norm,
        rng,
        ideal,
        None,
        worst,
    )
    # Expect at least one solution per principal direction
    associations, _ = associate(F_sel - F_sel.min(axis=0), ref_dirs_norm)
    assert set(associations) == set(range(n_obj))


def test_nsgaiii_survival_matches_backend_rank_path():
    rng = np.random.default_rng(12)
    X = rng.random((18, 5))
    F = rng.random((18, 3))
    X_off = rng.random((18, 5))
    F_off = rng.random((18, 3))
    ref_dirs = rng.random((18, 3))
    ref_dirs_norm = ref_dirs / np.linalg.norm(ref_dirs, axis=1, keepdims=True)
    ideal = np.full(3, np.inf)
    worst = np.full(3, -np.inf)

    expected = nsgaiii_survival(
        X,
        F,
        None,
        X_off,
        F_off,
        None,
        18,
        ref_dirs_norm,
        np.random.default_rng(7),
        ideal.copy(),
        None,
        worst.copy(),
        kernel=NumPyKernel(),
    )
    actual = nsgaiii_survival(
        X,
        F,
        None,
        X_off,
        F_off,
        None,
        18,
        ref_dirs_norm,
        np.random.default_rng(7),
        ideal.copy(),
        None,
        worst.copy(),
        kernel=NumbaKernel(),
    )

    for left, right in zip(expected[:6], actual[:6], strict=False):
        if left is None or right is None:
            assert left is right
        else:
            assert np.array_equal(left, right)


def test_nsgaiii_numba_same_seed_reproducible():
    cfg = _make_config(pop_size=15, divisions=4)
    problem = DTLZ2Problem(n_var=12, n_obj=3)

    result_a = NSGAIII(cfg, kernel=NumbaKernel()).run(problem, termination=("max_evaluations", 45), seed=3)
    result_b = NSGAIII(cfg, kernel=NumbaKernel()).run(problem, termination=("max_evaluations", 45), seed=3)

    assert np.array_equal(result_a["F"], result_b["F"])
    assert np.array_equal(result_a["X"], result_b["X"])


def _largest_complete_lattice(pop_size: int, n_obj: int) -> tuple[int, int]:
    # Das-Dennis lattice size, the cardinality of pymoo's get_reference_directions("das-dennis", M, n_partitions=H).
    divisions = max(p for p in range(1, pop_size + 1) if comb(p + n_obj - 1, n_obj - 1) <= pop_size)
    return divisions, comb(divisions + n_obj - 1, n_obj - 1)


def _lattice_points(weights: np.ndarray, divisions: int) -> set[tuple[int, ...]]:
    return {tuple(int(v) for v in row) for row in np.rint(weights * divisions)}


def _has_every_axis(directions: np.ndarray) -> bool:
    return all(np.any(np.all(np.isclose(directions, axis), axis=1)) for axis in np.eye(directions.shape[1]))


@pytest.mark.parametrize(("pop_size", "n_obj"), [(100, 3), (92, 3), (20, 3), (3, 3), (100, 2), (100, 5)])
def test_generated_reference_directions_are_complete_and_symmetric(pop_size: int, n_obj: int) -> None:
    divisions, size = _largest_complete_lattice(pop_size, n_obj)

    weights = _generate_weights(pop_size, n_obj, None)

    assert weights.shape == (size, n_obj)
    points = _lattice_points(weights, divisions)
    assert len(points) == size
    assert all(sum(point) == divisions for point in points)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert _has_every_axis(weights)
    for perm in permutations(range(n_obj)):
        assert _lattice_points(weights[:, list(perm)], divisions) == points


def test_cli_builder_nsgaiii_keeps_every_axis() -> None:
    problem = DTLZ2Problem(n_var=12, n_obj=3)
    algorithm, _ = build_algorithm("nsgaiii", "numpy", problem, ExperimentConfig())

    result = algorithm.run(problem, termination=("max_evaluations", 100), seed=0)

    directions = np.asarray(result["reference_directions"])
    assert directions.shape == (91, 3)
    assert _has_every_axis(directions)
    assert result["X"].shape[0] == 100


def test_nsgaiii_accepts_more_individuals_than_reference_directions() -> None:
    cfg = _make_config(pop_size=100, divisions=12)

    result = NSGAIII(cfg, kernel=NumPyKernel()).run(DTLZ2Problem(n_var=12, n_obj=3), termination=("max_evaluations", 300), seed=1)

    assert result["reference_directions"].shape == (91, 3)
    assert result["X"].shape[0] == 100
    assert result["evaluations"] == 300


def test_nsgaiii_rejects_fewer_individuals_than_reference_directions() -> None:
    cfg = _make_config(pop_size=90, divisions=12)

    with pytest.raises(ValueError, match="pop_size"):
        NSGAIII(cfg, kernel=NumPyKernel()).run(DTLZ2Problem(n_var=12, n_obj=3), termination=("max_evaluations", 200), seed=1)


def test_nsgaiii_rejects_an_oversized_lattice_before_generating_it(monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.engine.algorithm.nsgaiii.initialization as initialization

    def fail(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("the lattice was generated")

    monkeypatch.setattr(initialization, "load_or_generate_weight_vectors", fail)
    # 20 objectives and 12 divisions give comb(31, 19) = 141,120,525 directions, over 22 GB.
    cfg = _make_config(pop_size=100, divisions=12)

    with pytest.raises(ValueError, match="need pop_size >= 141120525"):
        NSGAIII(cfg, kernel=NumPyKernel()).run(DTLZ2Problem(n_var=29, n_obj=20), termination=("max_evaluations", 200), seed=1)


@pytest.mark.parametrize("algorithm", ["nsgaiii", "rvea"])
def test_nsgaiii_and_rvea_two_objective_defaults(algorithm: str) -> None:
    default = optimize("zdt1", algorithm=algorithm, max_evaluations=200, seed=1, engine="numpy")
    explicit = optimize("zdt1", algorithm=algorithm, pop_size=100, max_evaluations=200, seed=1, engine="numpy")

    assert default.meta["run_artifact_resolved_spec"]["population"]["initial_size"] == 100
    assert explicit.meta["run_artifact_resolved_spec"]["population"]["initial_size"] == 100
    if algorithm == "nsgaiii":
        assert explicit.data["reference_directions"].shape == (100, 2)
