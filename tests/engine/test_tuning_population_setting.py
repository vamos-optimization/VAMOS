"""``pop_size`` is a run setting: the grammars do not tune it, and the caller gives it with the assignment.

MOEA/D, NSGA-III and RVEA derive their populations from that fixed ``pop_size`` and the number of
objectives, as they derived them from the sampled one: MOEA/D runs the largest population, at most
``pop_size``, that has weight vectors, RVEA the largest complete Das-Dennis lattice with at most
``pop_size`` directions, and NSGA-III keeps ``pop_size`` with the largest lattice that fits in it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from vamos.engine.algorithm.components.weight_vectors import largest_weighted_population
from vamos.engine.reference_directions import largest_lattice_divisions, lattice_size
from vamos.engine.tuning import config_from_assignment
from vamos.engine.tuning.grammar import builtin_parameter_space, builtin_parameter_space_names
from vamos.engine.tuning.racing.bridge_spaces import build_moead_config_space

REFERENCE_SPACES = sorted(name for name in builtin_parameter_space_names() if name.split("_")[0] in {"moead", "nsgaiii", "rvea"})


@pytest.mark.parametrize("name", REFERENCE_SPACES)
@pytest.mark.parametrize("n_obj", [2, 3, 5])
@pytest.mark.parametrize("pop_size", [20, 60, 100, 150])
def test_reference_direction_algorithms_derive_their_population_from_the_fixed_size(name: str, n_obj: int, pop_size: int) -> None:
    space = builtin_parameter_space(name)
    config = config_from_assignment(name, {**space.default_values(), "pop_size": pop_size, "n_obj": n_obj})

    if space.algorithm == "moead":
        assert config.pop_size == (largest_weighted_population(pop_size, n_obj) if n_obj > 2 else pop_size)
    elif space.algorithm == "rvea":
        divisions = largest_lattice_divisions(pop_size, n_obj)
        assert (config.pop_size, config.n_partitions) == (lattice_size(n_obj, divisions), divisions)
    else:
        assert config.pop_size == pop_size


@pytest.mark.parametrize("name", sorted(builtin_parameter_space_names()))
def test_the_fixed_population_reaches_every_algorithm(name: str) -> None:
    space = builtin_parameter_space(name)
    config = config_from_assignment(name, {**space.default_values(), "pop_size": 48, "n_obj": 2})

    assert config.pop_size == 48


@pytest.mark.parametrize("name", ["nsgaii", "spea2", "smpso", "moead", "rvea"])
def test_an_assignment_without_a_population_runs_the_default_size(name: str) -> None:
    space = builtin_parameter_space(name)
    config = config_from_assignment(name, {**space.default_values(), "n_obj": 2})

    assert config.pop_size == 100


def _persist_moead_trial(output_dir: Path, pop_size: int, fixed_assignment: dict[str, int] | None = None) -> tuple[dict, dict]:
    from vamos.engine.tuning import Instance, TrialResult, TuningTask
    from vamos.experiment.cli._tune_post import persist_artifacts

    output_dir.mkdir()
    default = builtin_parameter_space("moead").default_values()
    task = TuningTask(
        name="moead",
        param_space=build_moead_config_space().to_param_space(),
        instances=[Instance(name="dtlz2", n_var=12, kwargs={})],
        seeds=[1],
        budget_per_run=300,
    )
    args = argparse.Namespace(
        backend="random",
        problem="dtlz2",
        algorithm="moead",
        n_var=12,
        n_obj=3,
        pop_size=pop_size,
        seed=1,
        aggregate_mode="mean",
        runtime_penalty=0.0,
        failure_score=0.0,
    )
    persist_artifacts(
        output_dir,
        args,
        task,
        dict(default),
        [TrialResult(0, dict(default), 0.5)],
        1.0,
        1,
        available_backends_fn=dict,
        fixed_assignment=fixed_assignment,
    )
    record = json.loads((output_dir / "tuning_history.json").read_text(encoding="utf-8"))[0]
    summary = json.loads((output_dir / "tuning_summary.json").read_text(encoding="utf-8"))
    return record, summary


def test_tune_records_the_population_moead_runs_instead_of_the_fixed_one(tmp_path: Path) -> None:
    # The trials hold no pop_size: the run's --pop-size is what MOEA/D replaces. 150 individuals
    # have no weight vectors with three objectives, so it runs 136, a complete lattice.
    record, summary = _persist_moead_trial(tmp_path / "adjusted", 150)
    assert record["details"]["effective"] == {"pop_size": 136}
    assert summary["best_config_effective"] == {"pop_size": 136}

    record, summary = _persist_moead_trial(tmp_path / "exact", 91)
    assert "effective" not in record["details"]
    assert summary["best_config_effective"] == {}


def test_tune_records_the_population_of_the_smoke_profile(tmp_path: Path) -> None:
    # --smoke fixes the population in its profile, which the evaluator sets in every configuration.
    record, summary = _persist_moead_trial(tmp_path / "smoke", 15, fixed_assignment={"pop_size": 150})
    assert record["details"]["effective"] == {"pop_size": 136}
    assert summary["best_config_effective"] == {"pop_size": 136}
