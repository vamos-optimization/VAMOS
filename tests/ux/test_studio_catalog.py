"""Tests for the Studio catalog, environment report and Run Experiment setups."""

from __future__ import annotations

import platform
from importlib import metadata
from pathlib import Path

import pytest

import vamos
from vamos.foundation.problem.registry import get_problem_specs
from vamos.ux.studio import environment
from vamos.ux.studio.algorithm_form import VAMOS_DEFAULT_CHOICE
from vamos.ux.studio.catalog import (
    algorithm_entries,
    filter_problems,
    has_reference_front,
    parameter_rows,
    population_rule,
    problem_entries,
    problem_rows,
    representative_problem,
)
from vamos.ux.studio.data import load_run_from_directory, load_runs_from_study
from vamos.ux.studio.experiment_setup import setup_from_run


def test_catalog_lists_every_registered_problem_with_its_default_dimensions() -> None:
    entries = {entry.key: entry for entry in problem_entries()}

    assert set(entries) == set(get_problem_specs())
    zdt1 = entries["zdt1"]
    assert (zdt1.encoding, zdt1.n_var, zdt1.n_obj, zdt1.n_obj_adjustable) == ("real", 30, 2, False)
    assert zdt1.reference_front
    assert "nsgaii" in zdt1.algorithms
    assert entries["dtlz2"].n_obj_adjustable
    assert all(row["Key"] in entries for row in problem_rows(list(entries.values())))


def test_problem_filters_combine_encoding_and_text() -> None:
    entries = problem_entries()

    binary = filter_problems(entries, encoding="binary")
    assert binary and all(entry.encoding == "binary" for entry in binary)
    assert [entry.key for entry in filter_problems(entries, text="ZDT1")] == ["zdt1"]
    assert filter_problems(entries, encoding="binary", text="zdt1") == []


def test_reference_front_must_have_the_number_of_objectives() -> None:
    assert has_reference_front("zdt1", 2)
    assert not has_reference_front("zdt1", 3)


def test_algorithms_list_the_encodings_they_have_operators_for() -> None:
    encodings = {entry.key: entry.encodings for entry in algorithm_entries()}

    assert encodings["nsgaii"] == ("real", "binary", "integer", "permutation", "mixed")
    assert "permutation" not in encodings["smpso"]
    assert representative_problem("permutation") is not None


def test_parameter_rows_show_vamos_defaults_and_allowed_values() -> None:
    zdt1 = next(entry for entry in problem_entries() if entry.key == "zdt1")

    rows = {str(row["Parameter"]).lstrip(" ↳ "): row for row in parameter_rows("nsgaii", zdt1)}

    assert "sbx" in str(rows["Crossover"]["VAMOS default"])
    assert "sbx" in str(rows["Crossover"]["Allowed values"])
    assert rows["Use external archive"]["VAMOS default"] == "no"
    assert any(str(row["Parameter"]).startswith("↳ ") for row in parameter_rows("nsgaii", zdt1))
    assert all(VAMOS_DEFAULT_CHOICE not in str(row["Allowed values"]) for row in rows.values())


def test_population_rule_follows_reference_directions() -> None:
    assert "reference directions" in population_rule("nsgaiii", 2)
    assert "weight vectors" in population_rule("moead", 3)
    assert "free setting" in population_rule("moead", 2)


def test_environment_report_reads_installed_metadata() -> None:
    report = environment.environment_report()

    assert report.python_version == platform.python_version()
    numpy_kernel = next(item for item in report.kernels if item.name == "numpy")
    assert numpy_kernel.available and numpy_kernel.detail == metadata.version("numpy")
    text = environment.environment_markdown(report)
    assert "**Kernels**" in text and "**Extras**" in text


def test_environment_report_names_the_extra_of_a_missing_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    installed = environment.installed_version
    monkeypatch.setattr(environment, "installed_version", lambda name: None if name in {"numba", "optuna"} else installed(name))

    report = environment.environment_report()

    numba = next(item for item in report.kernels if item.name == "numba")
    assert not numba.available and "compute" in numba.detail
    if report.vamos_version is not None:
        tuning = next(item for item in report.extras if item.name == "tuning")
        assert not tuning.available and "optuna" in tuning.detail


def test_every_listed_extra_is_declared_by_the_package() -> None:
    try:
        declared = set(metadata.metadata(environment.DISTRIBUTION).get_all("Provides-Extra") or [])
    except metadata.PackageNotFoundError:
        pytest.skip("VAMOS is not installed as a distribution.")

    assert set(environment.EXTRA_PURPOSES) <= declared


def test_setup_from_a_default_configuration_run(tmp_path: Path) -> None:
    spec = vamos.StudySpec(problems=["zdt1"], algorithms=["nsgaii"], seeds=[3], pop_size=8, max_evaluations=40, engine="numpy")
    study = vamos.create_study(spec, output=tmp_path / "study").run()
    (run,) = load_runs_from_study(study.root)

    setup = setup_from_run(run)

    assert (setup.problem, setup.n_var, setup.n_obj, setup.algorithm) == ("zdt1", 30, 2, "nsgaii")
    assert (setup.pop_size, setup.max_evaluations, setup.seed) == (8, 40, 3)
    assert "default configuration, as that run did" in " ".join(setup.notes)


def test_setup_from_a_customized_run_points_to_exact_reproduction(tmp_path: Path) -> None:
    from vamos.algorithms import NSGAIIConfig

    config = NSGAIIConfig.default(pop_size=8, n_var=6)
    result = vamos.optimize("zdt1", algorithm="nsgaii", algorithm_config=config, n_var=6, max_evaluations=24, seed=5, engine="numpy")
    stored = vamos.save_result(result, tmp_path / "run")
    run = load_run_from_directory(stored.root)

    setup = setup_from_run(run)

    assert (setup.problem, setup.n_var, setup.seed) == ("zdt1", 6, 5)
    notes = " ".join(setup.notes)
    assert "cannot load into the form" in notes
    assert f"vamos reproduce {stored.root.resolve()}" in notes
