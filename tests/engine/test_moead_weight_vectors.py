from __future__ import annotations

import numpy as np
import pytest

from vamos import optimize
from vamos.engine.algorithm.components.weight_vectors import load_or_generate_weight_vectors
from vamos.foundation.problem.dtlz import DTLZ2Problem


def test_moead_rejects_incompatible_das_dennis_lattice() -> None:
    with pytest.raises(ValueError, match="pop_size=40|requires pop_size"):
        load_or_generate_weight_vectors(40, 3, divisions=8, mode="jmetalpy")


def test_moead_accepts_exact_das_dennis_lattice() -> None:
    weights = load_or_generate_weight_vectors(45, 3, divisions=8, mode="jmetalpy")

    assert weights.shape == (45, 3)
    assert np.any(np.all(np.isclose(weights, [1.0, 0.0, 0.0]), axis=1))
    assert np.any(np.all(np.isclose(weights, [0.0, 1.0, 0.0]), axis=1))
    assert np.any(np.all(np.isclose(weights, [0.0, 0.0, 1.0]), axis=1))


def test_moead_weight_file_must_match_pop_size_exactly(tmp_path) -> None:
    path = tmp_path / "W3D_4.dat"
    np.savetxt(path, np.full((3, 3), 1.0 / 3.0))

    with pytest.raises(ValueError, match="Expected 4 weight vectors"):
        load_or_generate_weight_vectors(4, 3, path=str(path), mode="jmetalpy")


def test_optimize_moead_default_pop_size_is_lattice_compatible() -> None:
    problem = DTLZ2Problem(n_var=7, n_obj=3)

    result = optimize(problem, algorithm="moead", max_evaluations=91, seed=1, engine="numpy")

    assert result.meta["run_artifact_resolved_spec"]["population"]["initial_size"] == 91
    assert result.data["evaluations"] == 91


def test_optimize_moead_rejects_incompatible_explicit_pop_size() -> None:
    problem = DTLZ2Problem(n_var=7, n_obj=3)

    with pytest.raises(ValueError, match="Das-Dennis|pop_size"):
        optimize(problem, algorithm="moead", pop_size=40, max_evaluations=40, seed=1, engine="numpy")


def test_optimize_moead_three_objectives_pop_100_uses_packaged_weights() -> None:
    import numpy as np

    import vamos
    from vamos.engine.algorithm.components.weight_vectors import load_or_generate_weight_vectors
    from vamos.engine.algorithm.config import MOEADConfig
    from vamos.foundation.problem.dtlz import DTLZ2Problem
    from vamos.resources import weight_path

    config = MOEADConfig.default(pop_size=100, n_var=12, n_obj=3)
    assert config.weight_vectors == {"path": None, "divisions": None}
    weights = load_or_generate_weight_vectors(100, 3, mode="jmetalpy")
    np.testing.assert_allclose(weights, np.loadtxt(weight_path("W3D_100.dat")), atol=1e-12)

    result = vamos.optimize(DTLZ2Problem(n_var=12, n_obj=3), algorithm="moead", pop_size=100, max_evaluations=300, seed=1)
    assert result.F.shape[1] == 3


def test_moead_default_single_objective_returns() -> None:
    import threading

    from vamos.engine.algorithm.config import MOEADConfig

    result: dict[str, object] = {}
    worker = threading.Thread(target=lambda: result.update(config=MOEADConfig.default(pop_size=20, n_obj=1)), daemon=True)
    worker.start()
    worker.join(10)

    assert "config" in result, "MOEADConfig.default(pop_size=20, n_obj=1) did not return"
    assert result["config"].pop_size == 20  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("n_obj", "pop_size"),
    [
        (3, 100),
        (3, 500),
        (3, 600),
        (3, 800),
        (3, 1000),
        (3, 1200),
        (5, 1000),
        (5, 1200),
        (5, 1500),
        (5, 1800),
        (5, 2000),
        (5, 2500),
        (8, 156),
        (10, 275),
    ],
)
def test_packaged_weights_load_despite_their_rounding(n_obj: int, pop_size: int) -> None:
    import numpy as np

    from vamos.engine.algorithm.components.weight_vectors import load_or_generate_weight_vectors
    from vamos.resources import weight_path

    weights = load_or_generate_weight_vectors(pop_size, n_obj, mode="jmetalpy")

    stored = np.loadtxt(weight_path(f"W{n_obj}D_{pop_size}.dat"))
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(weights, stored, atol=1e-5)


def test_packaged_weights_with_another_size_are_not_used() -> None:
    from vamos.engine.algorithm.components.weight_vectors import load_or_generate_weight_vectors

    # W4D_100.dat holds 200 vectors: 100 individuals with four objectives keep the lattice error.
    with pytest.raises(ValueError, match="pop_size=100"):
        load_or_generate_weight_vectors(100, 4, mode="jmetalpy")
