from __future__ import annotations

import numpy as np
import pytest

from vamos import optimize
from vamos.engine.algorithm.components.weight_vectors import largest_weighted_population, load_or_generate_weight_vectors
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


def _packaged_weight_files() -> list[tuple[str, int, int]]:
    import re
    from importlib import resources

    names = (entry.name for entry in resources.files("vamos.resources").joinpath("weights").iterdir())
    matches = (re.fullmatch(r"W(\d+)D_(\d+)\.dat", name) for name in names)
    return sorted((match.group(0), int(match.group(1)), int(match.group(2))) for match in matches if match is not None)


def test_vamos_ships_the_jmetal_weight_files() -> None:
    files = _packaged_weight_files()

    # jMetal's 29 files, jMetalPy's W4D_100.dat and W3D_120.dat.
    assert len(files) == 31
    assert ("W4D_100.dat", 4, 100) in files and ("W4D_165.dat", 4, 165) in files


@pytest.mark.parametrize(("name", "n_obj", "pop_size"), _packaged_weight_files())
def test_every_packaged_weight_file_loads_with_one_vector_per_individual(name: str, n_obj: int, pop_size: int) -> None:
    from vamos.engine.algorithm.components.weight_vectors import _packaged_weights, read_weight_vectors
    from vamos.resources import weight_path

    packaged = _packaged_weights(n_obj, pop_size)
    read = read_weight_vectors(str(weight_path(name)), n_obj)

    assert packaged is not None and packaged.shape == (pop_size, n_obj)
    assert read.shape == (pop_size, n_obj)
    np.testing.assert_allclose(read.sum(axis=1), 1.0, atol=1e-5)
    assert read.min() >= 0.0
    np.testing.assert_allclose(packaged, read, atol=1e-5)


def test_moead_with_four_objectives_and_100_individuals_runs_the_packaged_weights() -> None:
    from vamos.resources import weight_path

    weights = load_or_generate_weight_vectors(100, 4, mode="jmetalpy")

    stored = np.loadtxt(weight_path("W4D_100.dat"))
    assert stored.shape == (100, 4) and len(np.unique(stored, axis=0)) == 100
    np.testing.assert_allclose(weights, stored, atol=1e-12)
    assert largest_weighted_population(100, 4) == 100
    assert largest_weighted_population(119, 4) == 100
    assert largest_weighted_population(99, 4) == 84


@pytest.mark.parametrize("pop_size", [91, 100, 136, 1000])
def test_the_packaged_directory_reads_as_any_jmetal_directory(pop_size: int) -> None:
    # The command line passes the packaged directory as weight_vectors.path: its files must load from there too,
    # W3D_136.dat with components of -5e-17 and W3D_1000.dat with rows that sum to 1 within 1e-6.
    from vamos.resources import weight_path

    directory = str(weight_path("W3D_91.dat").parent)

    weights = load_or_generate_weight_vectors(pop_size, 3, path=directory, mode="jmetalpy")

    assert weights.shape == (pop_size, 3) and weights.min() >= 0.0
    np.testing.assert_allclose(weights, np.loadtxt(weight_path(f"W3D_{pop_size}.dat")), atol=1e-12)


@pytest.mark.parametrize(("pop_size", "n_obj"), [(100, 4), (165, 4)])
def test_the_four_objective_files_load_from_the_packaged_directory(pop_size: int, n_obj: int) -> None:
    from vamos.resources import weight_path

    directory = str(weight_path("W3D_91.dat").parent)

    assert load_or_generate_weight_vectors(pop_size, n_obj, path=directory, mode="jmetalpy").shape == (pop_size, n_obj)


@pytest.mark.parametrize("separator", [" ", ",", ", ", "\t"])
def test_a_weight_file_may_separate_its_components_with_whitespace_or_commas(tmp_path, separator: str) -> None:
    from vamos.engine.algorithm.components.weight_vectors import read_weight_vectors
    from vamos.engine.reference_directions import das_dennis

    lattice = das_dennis(3, 4)
    path = tmp_path / "W3D_15.dat"
    path.write_text("\n".join(separator.join(repr(float(value)) for value in row) for row in lattice) + "\n", encoding="utf-8")

    np.testing.assert_array_equal(read_weight_vectors(str(path), 3), lattice)
    np.testing.assert_array_equal(load_or_generate_weight_vectors(15, 3, path=str(path), mode="jmetalpy"), lattice)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([[0.5, 0.5, 0.1]], "sum to 1"),
        ([[1.2, -0.2, 0.0]], "non-negative"),
        ([[0.5, 0.5]], "3 columns"),
        ([[float("nan"), 0.5, 0.5]], "finite"),
    ],
)
def test_a_weight_file_off_the_simplex_is_rejected(tmp_path, rows: list[list[float]], message: str) -> None:
    from vamos.engine.algorithm.components.weight_vectors import read_weight_vectors

    path = tmp_path / "weights.dat"
    np.savetxt(path, np.asarray(rows))

    with pytest.raises(ValueError, match=message):
        read_weight_vectors(str(path), 3)


def test_a_missing_weight_file_is_a_configuration_error(tmp_path) -> None:
    from vamos.engine.algorithm.components.weight_vectors import read_weight_vectors
    from vamos.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match="does not exist"):
        read_weight_vectors(str(tmp_path / "missing.dat"), 3)
    with pytest.raises(ConfigurationError, match="not a file"):
        read_weight_vectors(str(tmp_path), 3)


def test_largest_weighted_population_keeps_sizes_with_weights_and_lowers_the_others() -> None:
    assert largest_weighted_population(60, 3) == 55  # the complete lattice of 9 divisions
    assert largest_weighted_population(91, 3) == 91  # 12 divisions
    assert largest_weighted_population(102, 3) == 100  # the packaged W3D_100.dat
    assert largest_weighted_population(37, 2) == 37  # two objectives: any size
    with pytest.raises(ValueError, match="smallest Das-Dennis lattice"):
        largest_weighted_population(2, 3)


@pytest.mark.parametrize("n_obj", [3, 4, 5])
def test_largest_weighted_population_always_has_weight_vectors(n_obj: int) -> None:
    for pop_size in range(20, 201):
        size = largest_weighted_population(pop_size, n_obj)

        assert size <= pop_size and largest_weighted_population(size, n_obj) == size
        assert load_or_generate_weight_vectors(size, n_obj, mode="jmetalpy").shape == (size, n_obj)
