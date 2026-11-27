"""Riesz s-energy directions: any number of well-spaced vectors on the simplex, for MOEA/D, PPS, NSGA-III and RVEA.

Development after 1.5.2.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.algorithms import MOEADConfig, NSGAIIIConfig, PPSConfig, RVEAConfig
from vamos.engine.algorithm.components.weight_vectors import (
    describe_weight_vectors,
    load_or_generate_weight_vectors,
    read_weight_vectors,
    vector_digest,
    vector_source,
)
from vamos.engine.algorithm.moead import MOEAD
from vamos.engine.algorithm.nsgaiii import NSGAIII
from vamos.engine.algorithm.pps import PPS
from vamos.engine.algorithm.rvea import RVEA
from vamos.engine.algorithm.rvea.rvea import _generate_reference_vectors
from vamos.engine.reference_directions import RIESZ_ENERGY_MAX_POINTS, das_dennis, riesz_energy
from vamos.exceptions import ConfigurationError
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.dtlz import DTLZ2Problem
from vamos.resources import weight_path

ENERGY = {"method": "riesz_energy", "seed": 1}
FIELDS = {"moead": "weight_vectors", "pps": "weight_vectors", "nsgaiii": "reference_directions", "rvea": "reference_vectors"}


def _nearest(vectors: np.ndarray) -> np.ndarray:
    """Each vector's distance to its nearest other vector."""
    distances = np.linalg.norm(vectors[:, None] - vectors[None, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def _log_energy(vectors: np.ndarray, s: int) -> float:
    """The logarithm of the mean of ``1 / d_ij ** s`` over pairs, as pymoo's ``calc_potential_energy`` reports it."""
    distances = np.linalg.norm(vectors[:, None] - vectors[None, :], axis=2)[np.triu_indices(len(vectors), 1)]
    closest = distances.min()
    return float(-s * np.log(closest) + np.log(np.mean((closest / distances) ** s)))


# ---------------------------------------------------------------------------------------------------------------------
# The set


@pytest.mark.parametrize(("n_obj", "n_points"), [(2, 30), (3, 100), (3, 105), (5, 126), (8, 156)])
def test_the_set_has_n_points_on_the_simplex_with_every_corner(n_obj: int, n_points: int) -> None:
    vectors = riesz_energy(n_obj, n_points, seed=0)

    assert vectors.shape == (n_points, n_obj) and vectors.dtype == np.float64
    assert np.all(vectors >= 0.0)
    np.testing.assert_allclose(vectors.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    for corner in np.eye(n_obj):
        assert np.any(np.all(vectors == corner, axis=1)), corner
    assert _nearest(vectors).min() > 0.0
    # Rows in ascending lexicographic order, as das_dennis gives its lattice.
    order = np.lexsort(vectors.T[::-1])
    np.testing.assert_array_equal(order, np.arange(n_points))


def test_the_same_seed_gives_the_same_set_and_another_seed_another() -> None:
    from vamos.engine.reference_directions import _riesz_energy

    first = riesz_energy(3, 100, seed=7)
    _riesz_energy.cache_clear()
    again = riesz_energy(3, 100, seed=7)

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, riesz_energy(3, 100, seed=8))
    # A caller cannot change the set that later callers get.
    first[:] = 0.0
    np.testing.assert_array_equal(riesz_energy(3, 100, seed=7), again)


def test_as_many_points_as_objectives_are_the_corners() -> None:
    np.testing.assert_array_equal(riesz_energy(4, 4, seed=0), np.eye(4)[::-1])


@pytest.mark.parametrize(
    ("n_obj", "n_points", "kwargs", "message"),
    [
        (1, 5, {"seed": 0}, "n_obj"),
        (3, 2, {"seed": 0}, "n_points"),
        (3, 10, {"seed": -1}, "seed"),
        (3, 10, {"seed": True}, "seed"),
        (3, 10, {"seed": 0, "s": 0}, "s must"),
        (3, RIESZ_ENERGY_MAX_POINTS + 1, {"seed": 0}, "at most"),
    ],
)
def test_invalid_arguments_raise(n_obj: int, n_points: int, kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        riesz_energy(n_obj, n_points, **kwargs)


# Das-Dennis lattices at their sizes: every nearest-neighbour distance is sqrt(2) / divisions.
@pytest.mark.parametrize(("n_obj", "divisions", "share"), [(3, 12, 0.85), (3, 13, 0.85), (5, 5, 0.95)])
def test_at_a_lattice_size_the_closest_pair_is_nearly_as_far_apart_as_in_the_lattice(n_obj: int, divisions: int, share: float) -> None:
    lattice = das_dennis(n_obj, divisions)
    spacing = _nearest(lattice).min()

    for seed in range(3):
        vectors = riesz_energy(n_obj, lattice.shape[0], seed=seed)
        assert _nearest(vectors).min() >= share * spacing, seed
        # The lattice is a set of low energy: the Riesz set comes within 1% of its logarithm.
        assert _log_energy(vectors, n_obj**2) <= 1.01 * _log_energy(lattice, n_obj**2), seed


# pymoo 0.6.2, get_reference_directions("energy", n_obj, n_points, seed=1, 2 or 3): the same closest-pair distance
# and the same logarithm of the energy (s = n_obj ** 2) for the three seeds.
PYMOO = {(3, 100): (0.1011, 16.881), (3, 105): (0.1042, 16.981), (8, 156): (0.3310, 62.948)}


@pytest.mark.parametrize(("n_obj", "n_points"), sorted(PYMOO))
def test_the_set_is_as_well_spaced_as_pymoos(n_obj: int, n_points: int) -> None:
    closest, log_energy = PYMOO[(n_obj, n_points)]

    for seed in range(3):
        vectors = riesz_energy(n_obj, n_points, seed=seed)
        assert _nearest(vectors).min() >= 0.95 * closest, seed
        # At most 10% more energy (0.1 in its logarithm).
        assert _log_energy(vectors, n_obj**2) <= log_energy + 0.1, seed


def test_two_objectives_are_nearly_evenly_spaced() -> None:
    vectors = riesz_energy(2, 50, seed=0)

    assert _nearest(vectors).min() >= 0.95 * np.sqrt(2) / 49


# ---------------------------------------------------------------------------------------------------------------------
# The option


def _moead(pop_size: int, source: dict[str, Any]) -> MOEADConfig:
    builder = MOEADConfig.builder().pop_size(pop_size).neighbor_size(20).delta(0.9).replace_limit(2)
    builder = builder.crossover("sbx", prob=0.9, eta=20.0).mutation("pm", prob="1/n", eta=20.0).aggregation("tchebycheff")
    return builder.weight_vectors(**source).build()


def _pps(pop_size: int, source: dict[str, Any]) -> PPSConfig:
    builder = PPSConfig.builder().pop_size(pop_size).neighbor_size(20).crossover("de", cr=1.0, f=0.5)
    return builder.mutation("pm", prob="1/n", eta=20.0).weight_vectors(**source).build()


def _nsgaiii(pop_size: int, source: dict[str, Any]) -> NSGAIIIConfig:
    builder = NSGAIIIConfig.builder().pop_size(pop_size).crossover("sbx", prob=0.9, eta=30.0).mutation("pm", prob="1/n", eta=20.0)
    return builder.selection("tournament").reference_directions(**source).build()


def _rvea(pop_size: int, source: dict[str, Any]) -> RVEAConfig:
    builder = RVEAConfig.builder().pop_size(pop_size).reference_vectors(**source).alpha(2.0).adapt_freq(0.1)
    return builder.crossover("sbx", prob=0.9, eta=30.0).mutation("pm", prob="1/n", eta=20.0).build()


BUILD = {"moead": _moead, "pps": _pps, "nsgaiii": _nsgaiii, "rvea": _rvea}


def _start(algorithm: str, config: Any, n_obj: int) -> np.ndarray:
    """The vectors that an algorithm starts from, after its initial population is evaluated."""
    problem = DTLZ2Problem(n_var=n_obj + 9, n_obj=n_obj)
    termination = ("max_evaluations", 2 * config.pop_size)
    if algorithm == "moead":
        moead = MOEAD(config.to_dict(), kernel=NumPyKernel())
        moead.initialize(problem, termination, seed=1)
        assert moead._st is not None
        return np.asarray(moead._st.weights)
    if algorithm == "pps":
        pps = PPS(config.to_dict(), kernel=NumPyKernel())
        pps.initialize(problem, termination, seed=1)
        assert pps._st is not None
        return np.asarray(pps._st.weights)
    runner = (NSGAIII if algorithm == "nsgaiii" else RVEA)(config.to_dict(), kernel=NumPyKernel())
    runner.initialize(problem, termination, seed=1)
    assert runner.state is not None
    return np.asarray(runner.state.ref_dirs)


@pytest.mark.parametrize("algorithm", sorted(BUILD))
@pytest.mark.parametrize(("n_obj", "pop_size"), [(3, 100), (3, 105), (4, 50)])
def test_every_algorithm_starts_from_the_riesz_set_of_its_population(algorithm: str, n_obj: int, pop_size: int) -> None:
    config = BUILD[algorithm](pop_size, ENERGY)

    assert getattr(config, FIELDS[algorithm]) == {"path": None, "divisions": None, **ENERGY}
    np.testing.assert_array_equal(_start(algorithm, config, n_obj), riesz_energy(n_obj, pop_size, seed=1))


def test_the_seed_is_0_when_left_out() -> None:
    config = _moead(100, {"method": "riesz_energy"})

    assert config.weight_vectors == {"path": None, "divisions": None, "method": "riesz_energy"}
    np.testing.assert_array_equal(_start("moead", config, 3), riesz_energy(3, 100, seed=0))
    np.testing.assert_array_equal(load_or_generate_weight_vectors(100, 3, mode="auto", method="riesz_energy"), riesz_energy(3, 100, seed=0))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"method": "energy"}, "Unknown"),
        ({"method": "riesz_energy", "seed": -1}, "non-negative"),
        ({"method": "riesz_energy", "seed": 1.5}, "non-negative"),
        ({"seed": 3}, "needs method"),
        ({"method": "riesz_energy", "divisions": 12}, "excludes divisions"),
        ({"method": "riesz_energy", "path": "W3D_100.dat"}, "excludes path"),
    ],
)
@pytest.mark.parametrize("algorithm", sorted(BUILD))
def test_invalid_method_options_are_rejected_when_the_configuration_is_built(algorithm: str, value: Any, message: str) -> None:
    config = BUILD[algorithm](91, {"divisions": 12})

    with pytest.raises(ConfigurationError, match=message):
        dataclasses.replace(config, **{FIELDS[algorithm]: value})


def test_a_field_that_does_not_generate_vectors_rejects_a_method() -> None:
    assert vector_source({"divisions": 12, "method": None, "seed": None}) == {"divisions": 12}
    with pytest.raises(ConfigurationError, match="takes no method"):
        vector_source(ENERGY)
    with pytest.raises(ConfigurationError, match="excludes divisions, inner_divisions"):
        load_or_generate_weight_vectors(100, 3, divisions=3, inner_divisions=2, method="riesz_energy")


# ---------------------------------------------------------------------------------------------------------------------
# Defaults


def test_without_the_option_every_algorithm_starts_from_the_vectors_it_always_had() -> None:
    lattice = das_dennis(3, 12)
    jmetal_100 = read_weight_vectors(str(weight_path("W3D_100.dat")), 3)
    jmetal_100 = jmetal_100 / jmetal_100.sum(axis=1, keepdims=True)

    np.testing.assert_array_equal(_start("moead", MOEADConfig.default(pop_size=91, n_obj=3), 3), lattice)
    np.testing.assert_array_equal(_start("moead", MOEADConfig.default(pop_size=100, n_obj=3), 3), jmetal_100)
    np.testing.assert_array_equal(_start("moead", _moead(105, {}), 3), das_dennis(3, 13))
    np.testing.assert_array_equal(_start("pps", _pps(91, {}), 3), lattice)
    np.testing.assert_array_equal(_start("nsgaiii", NSGAIIIConfig.default(pop_size=92, n_obj=3), 3), lattice)
    np.testing.assert_array_equal(_start("nsgaiii", NSGAIIIConfig.default(pop_size=100, n_obj=3), 3), lattice)
    # RVEA builds its own lattice, which differs from das_dennis in the last bit of a few components.
    np.testing.assert_array_equal(_start("rvea", RVEAConfig.default(pop_size=91, n_obj=3), 3), _generate_reference_vectors(3, 12))
    np.testing.assert_array_equal(_start("rvea", _rvea(105, {"divisions": 13}), 3), _generate_reference_vectors(3, 13))
    for mode in ("auto", "exact", "jmetalpy"):
        np.testing.assert_array_equal(load_or_generate_weight_vectors(91, 3, mode=mode), lattice)


def test_without_the_option_configurations_serialize_as_before() -> None:
    for build, field in ((_moead, "weight_vectors"), (_pps, "weight_vectors"), (_nsgaiii, "reference_directions")):
        assert build(91, {"divisions": 12}).to_dict()[field] == {"path": None, "divisions": 12}
    assert _rvea(91, {"divisions": 12}).to_dict()["reference_vectors"] == {"path": None, "divisions": 12}
    assert "reference_vectors" not in RVEAConfig.default(n_obj=3).to_dict()
    assert MOEADConfig.default(n_obj=3).to_dict()["weight_vectors"] == {"path": None, "divisions": 12}
    assert NSGAIIIConfig.default(n_obj=3).to_dict()["reference_directions"] == {"path": None, "divisions": 12}
    assert describe_weight_vectors(91, 3) == {"source": "das_dennis", "divisions": 12, "vectors": 91}


# ---------------------------------------------------------------------------------------------------------------------
# Runs, records and studies


def _run(algorithm: str, config: Any, n_obj: int = 3) -> Any:
    return vamos.optimize(
        "dtlz2", algorithm=algorithm, algorithm_config=config, n_obj=n_obj, max_evaluations=2 * config.pop_size, seed=1, engine="numpy"
    )


@pytest.mark.parametrize("algorithm", sorted(BUILD))
def test_a_run_records_the_method_the_seed_and_the_digest_of_its_vectors(algorithm: str) -> None:
    resolved = _run(algorithm, BUILD[algorithm](100, ENERGY)).meta["run_artifact_resolved_spec"]["algorithm"]

    assert resolved["resolution"][FIELDS[algorithm]] == {
        "source": "riesz_energy",
        "seed": 1,
        "vectors": 100,
        "sha256": vector_digest(riesz_energy(3, 100, seed=1)),
    }


@pytest.mark.parametrize("algorithm", ["moead", "nsgaiii"])
def test_runs_with_riesz_vectors_verify_and_replay_exactly(algorithm: str, tmp_path: Path) -> None:
    source = vamos.save_result(_run(algorithm, BUILD[algorithm](100, ENERGY)), tmp_path / "source")

    assert vamos.verify_run(source.root).effective_replayability == "exact"
    assert vamos.reproduce(source.root, output=tmp_path / "replay").exact is True


def test_the_command_line_builders_read_the_method() -> None:
    from vamos.engine.algorithm.factory import build_algorithm
    from vamos.foundation.core.experiment_config import ExperimentConfig

    problem = DTLZ2Problem(n_var=12, n_obj=3)
    config = ExperimentConfig(population_size=100)
    for algorithm, field in (("moead", "weight_vectors"), ("nsgaiii", "reference_directions"), ("rvea", "reference_vectors")):
        _, resolved = build_algorithm(algorithm, "numpy", problem, config, variations={algorithm: {field: dict(ENERGY)}})
        assert getattr(resolved, field)["method"] == "riesz_energy" and getattr(resolved, field)["seed"] == 1


@pytest.mark.parametrize("algorithm", sorted(BUILD))
def test_a_study_plans_riesz_vectors_for_a_population_without_a_lattice(algorithm: str) -> None:
    spec = vamos.StudySpec(
        problems=["dtlz2"],
        algorithms=[algorithm],
        seeds=[0],
        pop_size=100,
        n_obj=3,
        max_evaluations=200,
        engine="numpy",
        algorithm_configs={algorithm: {FIELDS[algorithm]: dict(ENERGY)}},
    )

    resolved = vamos.plan_study(spec).plan.tasks[0].resolved_run_spec["algorithm"]

    assert resolved["config"][FIELDS[algorithm]] == ENERGY
    assert resolved["resolution"][FIELDS[algorithm]]["sha256"] == vector_digest(riesz_energy(3, 100, seed=1))


@pytest.mark.parametrize("algorithm", ["nsgaiii", "rvea"])
def test_study_planning_rejects_a_method_with_a_lattice(algorithm: str) -> None:
    from vamos.experiment.study.errors import UnresolvedStudyTaskError

    spec = vamos.StudySpec(
        problems=["dtlz2"],
        algorithms=[algorithm],
        seeds=[0],
        pop_size=91,
        n_obj=3,
        max_evaluations=182,
        engine="numpy",
        algorithm_configs={algorithm: {FIELDS[algorithm]: {"method": "riesz_energy", "divisions": 12}}},
    )

    with pytest.raises(UnresolvedStudyTaskError, match="INVALID_REFERENCE_DIRECTIONS"):
        vamos.plan_study(spec)
