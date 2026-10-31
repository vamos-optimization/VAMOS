"""Tests for the Panel-free algorithm configuration forms of VAMOS Studio."""

from __future__ import annotations

import pytest

from vamos.engine.tuning.racing.param_space import Real
from vamos.foundation.problem.registry import make_problem_selection
from vamos.ux.studio.algorithm_form import (
    VAMOS_DEFAULT_CHOICE,
    AlgorithmForm,
    FormParameter,
    _form_parameter,
    algorithms_for_encoding,
    build_algorithm_form,
    config_fields_for,
    default_algorithm_config,
    population_is_derived,
    studio_algorithms,
)
from vamos.ux.studio.services import _run_algorithm

ENCODINGS = ("real", "binary", "integer", "permutation", "mixed")
FORM_CASES = [(algorithm, encoding) for encoding in ENCODINGS for algorithm in studio_algorithms()]


def _form(algorithm: str, encoding: str = "real", *, n_obj: int = 2) -> AlgorithmForm:
    form = build_algorithm_form(algorithm, encoding=encoding, n_var=12, n_obj=n_obj, pop_size=40)
    assert form is not None
    return form


def _alternative(parameter: FormParameter) -> object | None:
    if parameter.kind == "boolean":
        return not parameter.start
    if parameter.kind == "categorical":
        choices = [choice for choice in parameter.choices if choice not in (parameter.start, VAMOS_DEFAULT_CHOICE)]
        return choices[-1] if choices else None
    assert parameter.low is not None and parameter.high is not None
    value = parameter.high if parameter.start != parameter.high else parameter.low
    return int(value) if parameter.kind == "int" else value


def test_every_enabled_algorithm_has_a_real_encoding_form() -> None:
    assert set(algorithms_for_encoding("real")) == set(studio_algorithms())


@pytest.mark.parametrize(("algorithm", "encoding"), FORM_CASES)
def test_untouched_form_resolves_to_the_exact_default_configuration(algorithm: str, encoding: str) -> None:
    form = build_algorithm_form(algorithm, encoding=encoding, n_var=12, n_obj=2, pop_size=40)
    if form is None:
        pytest.skip(f"{algorithm} has no {encoding} configuration space")

    resolved = form.resolve(form.start_values())

    assert resolved.config is form.baseline
    assert resolved.differences == ()


@pytest.mark.parametrize(("algorithm", "encoding"), FORM_CASES)
def test_every_offered_control_changes_the_configuration(algorithm: str, encoding: str) -> None:
    form = build_algorithm_form(algorithm, encoding=encoding, n_var=12, n_obj=2, pop_size=40)
    if form is None:
        pytest.skip(f"{algorithm} has no {encoding} configuration space")
    start = form.start_values()
    active = form.active_names(start)

    for parameter in form.parameters:
        alternative = _alternative(parameter)
        if parameter.name not in active or alternative is None:
            continue
        resolved = form.resolve({**start, parameter.name: alternative})

        assert resolved.changed == (parameter.name,)
        assert resolved.unapplied == ()
        assert resolved.differences, f"{algorithm}/{encoding}: changing {parameter.name} had no effect"


def test_a_change_keeps_every_other_default_field() -> None:
    form = _form("smsemoa")
    baseline = form.baseline.to_dict()

    resolved = form.resolve({**form.start_values(), "crossover_prob": 0.75})
    config = resolved.config.to_dict()

    assert [difference.field for difference in resolved.differences] == ["crossover"]
    assert config["crossover"] == ("sbx", {"prob": 0.75, "eta": 20.0})
    assert config["selection"] == baseline["selection"]
    assert config["reference_point"] == baseline["reference_point"]


def test_conditional_parameters_follow_their_parent() -> None:
    form = _form("nsgaii")
    start = form.start_values()

    assert "crossover_eta" in form.active_names(start)
    assert "crossover_alpha" not in form.active_names(start)
    switched = form.active_names({**start, "crossover": "blx_alpha"})
    assert "crossover_alpha" in switched
    assert "crossover_eta" not in switched


def test_defaults_outside_the_tuning_range_are_kept_and_explained() -> None:
    # No built-in default lies outside its tuning range since IBEA's kappa default became 0.05.
    kappa = _form_parameter(Real("kappa", 0.01, 0.2), 0, {"kappa": 1.0})

    assert kappa.start == pytest.approx(1.0)
    assert kappa.high is not None and kappa.high >= 1.0
    assert "outside the tuning range" in kappa.note


def test_ibea_kappa_default_lies_in_its_tuning_range() -> None:
    kappa = next(parameter for parameter in _form("ibea").parameters if parameter.name == "kappa")

    assert kappa.start == pytest.approx(0.05)
    assert kappa.note == ""


def test_population_follows_reference_directions_where_required() -> None:
    assert population_is_derived("nsgaiii", 2)
    assert population_is_derived("rvea", 3)
    assert population_is_derived("moead", 3)
    assert not population_is_derived("moead", 2)
    assert not population_is_derived("nsgaii", 3)


@pytest.mark.parametrize(("problem_name", "algorithm"), [("zdt1", "nsgaiii"), ("zdt1", "rvea"), ("dtlz2", "moead")])
def test_default_configuration_runs_with_the_studio_population_size(problem_name: str, algorithm: str) -> None:
    problem = make_problem_selection(problem_name).instantiate()
    config = default_algorithm_config(algorithm, encoding="real", n_var=problem.n_var, n_obj=problem.n_obj, pop_size=100)

    result = _run_algorithm(
        problem, algorithm=algorithm, algorithm_config=config, termination=("max_evaluations", 300), seed=0, engine="numpy"
    )

    assert result["F"].shape[1] == problem.n_obj


def test_binary_problems_start_from_binary_operators_that_can_be_replaced() -> None:
    problem = make_problem_selection("zdt5").instantiate()
    form = build_algorithm_form("nsgaiii", encoding=problem.encoding, n_var=problem.n_var, n_obj=problem.n_obj, pop_size=100)
    assert form is not None
    crossover = next(parameter for parameter in form.parameters if parameter.name == "crossover")
    # The default configuration follows the encoding, so the form starts from a binary crossover.
    assert crossover.start == "uniform"

    resolved = form.resolve({**form.start_values(), "crossover": "hux", "mutation": "bitflip"})
    result = _run_algorithm(
        problem, algorithm="nsgaiii", algorithm_config=resolved.config, termination=("max_evaluations", 100), seed=0, engine="numpy"
    )

    assert resolved.config.to_dict()["crossover"][0] == "hux"
    assert result["F"].shape[1] == 2


@pytest.mark.parametrize(
    ("parameter", "fields"),
    [
        ("crossover_eta", ("crossover",)),
        ("de_f", ("crossover",)),
        ("gaussian_sigma", ("mutation",)),
        ("mutation_prob_factor", ("mutation_prob_factor",)),
        ("selection_pressure", ("selection",)),
        ("archive_prune_policy", ("external_archive",)),
        ("blx_repair", ("repair",)),
        ("n_partitions", ("n_partitions", "pop_size")),
        ("kappa", ("kappa",)),
    ],
)
def test_space_parameters_map_to_configuration_fields(parameter: str, fields: tuple[str, ...]) -> None:
    assert config_fields_for(parameter) == fields
