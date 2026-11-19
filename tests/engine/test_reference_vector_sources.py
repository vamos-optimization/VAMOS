"""MOEA/D, NSGA-III and RVEA take their vectors from one kind of option: a file, a directory or a lattice."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.algorithms import MOEADConfig, NSGAIIIConfig, RVEAConfig
from vamos.engine.algorithm.components.weight_vectors import read_weight_vectors
from vamos.engine.algorithm.moead import MOEAD
from vamos.engine.algorithm.nsgaiii import NSGAIII
from vamos.engine.algorithm.rvea import RVEA
from vamos.engine.reference_directions import das_dennis, two_layer_lattice, two_layer_size
from vamos.exceptions import ConfigurationError
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.dtlz import DTLZ2Problem
from vamos.resources import weight_path

Source = dict[str, Any]


def _moead(pop_size: int, source: Source) -> MOEADConfig:
    return (
        MOEADConfig.builder()
        .pop_size(pop_size)
        .neighbor_size(20)
        .delta(0.9)
        .replace_limit(2)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .aggregation("tchebycheff")
        .weight_vectors(**source)
        .build()
    )


def _nsgaiii(pop_size: int, source: Source) -> NSGAIIIConfig:
    return (
        NSGAIIIConfig.builder()
        .pop_size(pop_size)
        .crossover("sbx", prob=0.9, eta=30.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament")
        .reference_directions(**source)
        .build()
    )


def _rvea(pop_size: int, source: Source) -> RVEAConfig:
    return (
        RVEAConfig.builder()
        .pop_size(pop_size)
        .reference_vectors(**source)
        .alpha(2.0)
        .adapt_freq(0.1)
        .crossover("sbx", prob=0.9, eta=30.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .build()
    )


def _start(algorithm: str, pop_size: int, n_obj: int, source: Source) -> np.ndarray:
    """The vectors each algorithm starts from, after its initial population is evaluated."""
    problem = DTLZ2Problem(n_var=n_obj + 9, n_obj=n_obj)
    termination = ("max_evaluations", 2 * pop_size)
    kernel = NumPyKernel()
    if algorithm == "moead":
        moead = MOEAD(_moead(pop_size, source).to_dict(), kernel=kernel)
        moead.initialize(problem, termination, seed=1)
        assert moead._st is not None
        return np.asarray(moead._st.weights)
    if algorithm == "nsgaiii":
        nsgaiii = NSGAIII(_nsgaiii(pop_size, source).to_dict(), kernel=kernel)
        nsgaiii.initialize(problem, termination, seed=1)
        assert nsgaiii.state is not None
        return np.asarray(nsgaiii.state.ref_dirs)
    rvea = RVEA(_rvea(pop_size, source).to_dict(), kernel=kernel)
    rvea.initialize(problem, termination, seed=1)
    assert rvea.state is not None
    return np.asarray(rvea.state.ref_dirs)


ALGORITHMS = ("moead", "nsgaiii", "rvea")


@pytest.mark.parametrize(("n_obj", "pop_size"), [(3, 100), (8, 156)])
@pytest.mark.parametrize("by", ["file", "directory"])
def test_the_three_algorithms_start_from_the_same_jmetal_file(n_obj: int, pop_size: int, by: str) -> None:
    file = weight_path(f"W{n_obj}D_{pop_size}.dat")
    path = str(file) if by == "file" else str(file.parent)
    expected = read_weight_vectors(str(file), n_obj)

    for algorithm in ALGORITHMS:
        np.testing.assert_array_equal(_start(algorithm, pop_size, n_obj, {"path": path}), expected, err_msg=algorithm)


def test_the_three_algorithms_start_from_the_same_two_layer_lattice() -> None:
    expected = two_layer_lattice(8, 3, 2)

    for algorithm in ALGORITHMS:
        np.testing.assert_array_equal(_start(algorithm, 156, 8, {"divisions": 3, "inner_divisions": 2}), expected, err_msg=algorithm)


def test_a_two_layer_lattice_of_eight_objectives_has_the_size_of_jmetals_file() -> None:
    lattice = two_layer_lattice(8, 3, 2)

    assert two_layer_size(8, 3, 2) == 156 == len(lattice) == len(np.loadtxt(weight_path("W8D_156.dat")))
    np.testing.assert_allclose(lattice.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_array_equal(lattice[:120], das_dennis(8, 3))
    np.testing.assert_allclose(lattice[120:], das_dennis(8, 2) / 2 + 1 / 16, atol=1e-15)
    # The inside layer lies off the boundary; jMetal's file stacks the two lattices unshrunk, so it repeats the axes.
    assert lattice[120:].min() >= 1 / 16 - 1e-15
    assert len(np.unique(lattice, axis=0)) == 156


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_a_missing_path_raises_before_any_evaluation(algorithm: str, tmp_path: Path) -> None:
    missing = tmp_path / "missing" / "W3D_91.dat"

    with pytest.raises(ConfigurationError, match="does not exist"):
        _start(algorithm, 91, 3, {"path": str(missing)})
    assert not missing.parent.exists()


@pytest.mark.parametrize("algorithm", ["nsgaiii", "rvea"])
def test_a_directory_without_the_file_raises(algorithm: str, tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match=r"W3D_91\.dat"):
        _start(algorithm, 91, 3, {"path": str(tmp_path)})


def test_moead_keeps_saving_the_lattice_in_a_directory_without_the_file(tmp_path: Path) -> None:
    with pytest.warns(RuntimeWarning, match="Generated and saved"):
        weights = _start("moead", 91, 3, {"path": str(tmp_path)})

    np.testing.assert_array_equal(np.loadtxt(tmp_path / "W3D_91.dat"), weights)


@pytest.mark.parametrize("algorithm", ["moead", "rvea"])
def test_a_file_with_another_number_of_vectors_raises(algorithm: str) -> None:
    with pytest.raises(ValueError, match="100"):
        _start(algorithm, 91, 3, {"path": str(weight_path("W3D_100.dat"))})


def test_rvea_runs_100_individuals_with_three_objectives_from_jmetals_file() -> None:
    config = _rvea(100, {"path": str(weight_path("W3D_100.dat"))})

    result = vamos.optimize("dtlz2", algorithm="rvea", algorithm_config=config, n_obj=3, max_evaluations=500, seed=1, engine="numpy")

    assert result.data["evaluations"] == 500
    assert result.meta["run_artifact_resolved_spec"]["population"]["initial_size"] == 100
    assert config.n_partitions == 12  # not used with reference_vectors


def test_rvea_divisions_are_the_lattice_of_n_partitions() -> None:
    by_partitions = RVEAConfig.default(pop_size=91, n_obj=3)
    by_vectors = dataclasses.replace(by_partitions, reference_vectors={"divisions": 12})

    first = vamos.optimize("dtlz2", algorithm="rvea", algorithm_config=by_partitions, max_evaluations=300, seed=3, engine="numpy")
    second = vamos.optimize("dtlz2", algorithm="rvea", algorithm_config=by_vectors, max_evaluations=300, seed=3, engine="numpy")

    np.testing.assert_array_equal(first.F, second.F)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"path": None, "divisons": 3}, "divisons"),
        ({"divisions": 0}, "positive integer"),
        ({"divisions": True}, "positive integer"),
        ({"inner_divisions": 2}, "needs divisions"),
        ({"path": "W3D_100.dat", "divisions": 3, "inner_divisions": 2}, "file or a lattice"),
        ({"path": 3}, "path"),
        (12, "mapping"),
    ],
)
@pytest.mark.parametrize("build", [_moead, _nsgaiii, _rvea])
def test_invalid_vector_options_are_rejected_when_the_configuration_is_built(build: Any, value: Any, message: str) -> None:
    config = build(91, {"divisions": 12})
    field = {"MOEADConfig": "weight_vectors", "NSGAIIIConfig": "reference_directions", "RVEAConfig": "reference_vectors"}

    with pytest.raises(ConfigurationError, match=message):
        dataclasses.replace(config, **{field[type(config).__name__]: value})


def test_a_path_string_is_a_path_and_configurations_round_trip(tmp_path: Path) -> None:
    path = str(weight_path("W3D_100.dat"))
    rvea = dataclasses.replace(RVEAConfig.default(pop_size=91, n_obj=3), reference_vectors=path)

    assert rvea.reference_vectors == {"path": path, "divisions": None}
    for config in (rvea, _moead(156, {"divisions": 3, "inner_divisions": 2}), _nsgaiii(156, {"divisions": 3, "inner_divisions": 2})):
        assert type(config).from_dict(config.to_dict()) == config


def test_configurations_without_the_new_options_serialize_as_before() -> None:
    # VAMOS 1.4.0's to_dict(): no reference_vectors for RVEA, no inner_divisions in the other two.
    assert "reference_vectors" not in RVEAConfig.default(n_obj=3).to_dict()
    assert NSGAIIIConfig.default(n_obj=3).to_dict()["reference_directions"] == {"path": None, "divisions": 12}
    assert MOEADConfig.default(n_obj=3).to_dict()["weight_vectors"] == {"path": None, "divisions": 12}
    assert NSGAIIIConfig.builder().pop_size(10).crossover("sbx").mutation("pm").selection("tournament").build().reference_directions == {}


def _resolved(algorithm: str, config: Any, n_obj: int, budget: int) -> dict[str, Any]:
    result = vamos.optimize(
        "dtlz2", algorithm=algorithm, algorithm_config=config, n_obj=n_obj, max_evaluations=budget, seed=1, engine="numpy"
    )
    return dict(result.meta["run_artifact_resolved_spec"]["algorithm"]["resolution"])


def test_the_resolved_spec_records_a_file_and_a_two_layer_lattice() -> None:
    path = str(weight_path("W3D_100.dat"))

    assert _resolved("nsgaiii", _nsgaiii(100, {"path": path}), 3, 200)["reference_directions"] == {"source": "file", "file": "W3D_100.dat"}
    assert _resolved("rvea", _rvea(100, {"path": path}), 3, 200)["reference_vectors"] == {
        "source": "file",
        "file": "W3D_100.dat",
        "vectors": 100,
    }
    assert _resolved("moead", _moead(156, {"divisions": 3, "inner_divisions": 2}), 8, 312)["weight_vectors"] == {
        "source": "two_layer",
        "divisions": 3,
        "inner_divisions": 2,
        "vectors": 156,
    }
    # A lattice is stated by the configuration: NSGA-III and RVEA keep the resolution of VAMOS 1.4.0.
    assert _resolved("nsgaiii", NSGAIIIConfig.default(pop_size=91, n_obj=3), 3, 182) == {"contract_version": 1}
    assert _resolved("rvea", RVEAConfig.default(pop_size=91, n_obj=3), 3, 182) == {"contract_version": 1}


@pytest.mark.parametrize(
    ("algorithm", "config", "n_obj"),
    [
        ("rvea", _rvea(100, {"path": str(weight_path("W3D_100.dat"))}), 3),
        ("nsgaiii", _nsgaiii(100, {"path": str(weight_path("W3D_91.dat").parent)}), 3),
        ("moead", _moead(156, {"divisions": 3, "inner_divisions": 2}), 8),
        ("rvea", _rvea(156, {"divisions": 3, "inner_divisions": 2}), 8),
    ],
)
def test_runs_with_the_new_options_replay_exactly(algorithm: str, config: Any, n_obj: int, tmp_path: Path) -> None:
    result = vamos.optimize("dtlz2", algorithm=algorithm, algorithm_config=config, n_obj=n_obj, max_evaluations=2 * config.pop_size, seed=1)
    source = vamos.save_result(result, tmp_path / "source")

    assert vamos.verify_run(source.root).effective_replayability == "exact"
    assert vamos.reproduce(source.root, output=tmp_path / "replay").exact is True


def test_the_command_line_builder_reads_rvea_reference_vectors() -> None:
    from vamos.engine.algorithm.factory import build_algorithm
    from vamos.foundation.core.experiment_config import ExperimentConfig

    problem = DTLZ2Problem(n_var=12, n_obj=3)
    config = ExperimentConfig(population_size=100)
    variations = {"rvea": {"reference_vectors": str(weight_path("W3D_100.dat"))}}

    algorithm, resolved = build_algorithm("rvea", "numpy", problem, config, variations=variations)
    result = algorithm.run(problem, termination=("max_evaluations", 200), seed=1)

    assert resolved.reference_vectors == {"path": str(weight_path("W3D_100.dat")), "divisions": None}
    assert result["evaluations"] == 200


def test_study_planning_checks_rvea_reference_vectors() -> None:
    def plan(pop_size: int, config: dict[str, Any]) -> Any:
        spec = vamos.StudySpec(
            problems=["dtlz2"],
            algorithms=["rvea"],
            seeds=[0],
            pop_size=pop_size,
            n_obj=8,
            max_evaluations=2 * pop_size,
            engine="numpy",
            algorithm_configs={"rvea": config},
        )
        return vamos.plan_study(spec)

    from vamos.experiment.study.errors import UnresolvedStudyTaskError

    planned = plan(156, {"reference_vectors": {"divisions": 3, "inner_divisions": 2}})
    assert planned.valid
    assert planned.plan.tasks[0].resolved_run_spec["population"]["initial_size"] == 156
    with pytest.raises(UnresolvedStudyTaskError, match="REFERENCE_DIRECTION_POPULATION_MISMATCH"):
        plan(150, {"reference_vectors": {"divisions": 3, "inner_divisions": 2}})
    with pytest.raises(UnresolvedStudyTaskError, match="UNSAFE_REFERENCE_DIRECTIONS"):
        plan(156, {"reference_vectors": {"path": str(weight_path("W8D_156.dat"))}})


def test_study_planning_checks_an_nsgaiii_two_layer_lattice() -> None:
    from vamos.experiment.study.errors import UnresolvedStudyTaskError

    def plan(pop_size: int) -> Any:
        directions = {"path": None, "divisions": 3, "inner_divisions": 2}
        spec = vamos.StudySpec(
            problems=["dtlz2"],
            algorithms=["nsgaiii"],
            seeds=[0],
            pop_size=pop_size,
            n_obj=8,
            max_evaluations=2 * pop_size,
            engine="numpy",
            algorithm_configs={"nsgaiii": {"reference_directions": directions}},
        )
        return vamos.plan_study(spec)

    assert plan(160).valid
    with pytest.raises(UnresolvedStudyTaskError, match=">= 156"):
        plan(150)
