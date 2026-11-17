"""Tests for the Panel-free algorithm configuration forms of VAMOS Studio."""

from __future__ import annotations

import json

import pytest

from vamos.engine.tuning.grammar import Parameter, ParameterSpace
from vamos.engine.tuning.racing.bridge_assignments import config_from_assignment
from vamos.foundation.problem.registry import make_problem_selection
from vamos.ux.studio.algorithm_form import (
    AlgorithmForm,
    FormParameter,
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
        choices = [choice for choice in parameter.choices if choice != parameter.start]
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

    resolved = form.resolve({**form.start_values(), "crossover_probability": 0.75})
    config = resolved.config.to_dict()

    assert [difference.field for difference in resolved.differences] == ["crossover"]
    assert config["crossover"] == ("sbx", {"prob": 0.75, "eta": 20.0})
    assert config["selection"] == baseline["selection"]
    assert config["reference_point"] == baseline["reference_point"]


def test_conditional_parameters_follow_their_parent() -> None:
    form = _form("nsgaii")
    start = form.start_values()

    assert "sbx_distribution_index" in form.active_names(start)
    assert "crossover_alpha" not in form.active_names(start)
    switched = form.active_names({**start, "crossover": "blx_alpha"})
    assert "crossover_alpha" in switched
    assert "sbx_distribution_index" not in switched


def _depths(grammar: ParameterSpace) -> dict[str, int]:
    depths: dict[str, int] = {}

    def visit(parameters: tuple[Parameter, ...], depth: int) -> None:
        for parameter in parameters:
            depths[parameter.name] = depth
            visit(parameter.global_sub_parameters, depth + 1)
            for production in parameter.productions:
                visit(production, depth + 1)

    visit(grammar.parameters, 0)
    return depths


@pytest.mark.parametrize(("algorithm", "encoding"), FORM_CASES)
def test_a_form_is_its_grammar(algorithm: str, encoding: str) -> None:
    # The form shows the grammar's tree, starts from its defaults, and edits each parameter within the one
    # range that tuning uses (no wider range for editing by hand).
    form = build_algorithm_form(algorithm, encoding=encoding, n_var=12, n_obj=2, pop_size=40)
    if form is None:
        pytest.skip(f"{algorithm} has no {encoding} configuration space")
    grammar = form.grammar
    order = [parameter.name for parameter in grammar.walk()]
    depths = _depths(grammar)

    assert [parameter.name for parameter in form.parameters] == [name for name in order if name in form.start_values()]
    for parameter in form.parameters:
        node = grammar.parameter(parameter.name)
        assert parameter.depth == depths[node.name]
        if parameter.name in POPULATION_DEFAULTS.get(algorithm, ()):
            # The grammar's default is that of 100 individuals; the form starts at this run's.
            assert parameter.start == form.baseline.to_dict()[parameter.name] == 40 and "population" in parameter.note
        else:
            assert parameter.start == node.default and parameter.note == ""
        if node.type == "categorical":
            assert parameter.choices == node.values
        elif node.range is not None:
            assert (parameter.low, parameter.high, parameter.log) == (*node.range, node.log)


# The parameters whose default configuration follows the population; the grammars give the value of 100 individuals.
POPULATION_DEFAULTS = {"spea2": ("archive_size",), "smpso": ("archive_size",)}


@pytest.mark.parametrize("pop_size", [12, 40, 150])
@pytest.mark.parametrize(("algorithm", "encoding"), FORM_CASES)
def test_an_untouched_form_shows_what_it_runs(algorithm: str, encoding: str, pop_size: int) -> None:
    # Building the starting values gives the default configuration of the run's population, field by
    # field, for every population: no control shows a value that the run does not use.
    form = build_algorithm_form(algorithm, encoding=encoding, n_var=12, n_obj=2, pop_size=pop_size)
    if form is None:
        pytest.skip(f"{algorithm} has no {encoding} configuration space")
    reference = build_algorithm_form(algorithm, encoding=encoding, n_var=12, n_obj=2, pop_size=100)
    assert reference is not None

    assert _start_differences(form) <= _start_differences(reference)


def _start_differences(form: AlgorithmForm) -> set[str]:
    """The configuration fields that the starting values build differently from the baseline."""
    baseline = form.baseline.to_dict()
    assignment = {name: form.start_values()[name] for name in form.active_names(form.start_values())}
    built = config_from_assignment(form.grammar.variant, {**assignment, "pop_size": baseline["pop_size"], "n_obj": form.n_obj}).to_dict()
    return {
        key for key in set(baseline) | set(built) if json.dumps(baseline.get(key), default=str) != json.dumps(built.get(key), default=str)
    }


@pytest.mark.parametrize("algorithm", sorted(POPULATION_DEFAULTS))
def test_the_default_of_100_individuals_is_a_change_for_another_population(algorithm: str) -> None:
    form = _form(algorithm)  # 40 individuals
    resolved = form.resolve({**form.start_values(), "archive_size": 100})

    assert resolved.changed == ("archive_size",)
    assert resolved.config.to_dict()["archive_size"] == 100


def test_forms_start_from_the_automatic_initializer_and_repair() -> None:
    parameters = {parameter.name: parameter for parameter in _form("nsgaii").parameters}

    for name in ("initial_solutions", "repair"):
        assert parameters[name].start == "auto" and parameters[name].choices[0] == "auto"


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


def test_cmoeams_de_variant_reaches_its_configuration_field() -> None:
    # CMOEA-MS keeps its DE variant in its own field, not in the crossover tuple.
    form = _form("cmoeams")

    resolved = form.resolve({**form.start_values(), "variation": "differential_evolution", "de_variant": "rand_1_exp"})

    assert resolved.config.to_dict()["crossover"][0] == "de"
    assert resolved.config.to_dict()["de_variant"] == "rand_1_exp"


@pytest.mark.parametrize(
    ("parameter", "fields"),
    [
        ("crossover_eta", ("crossover",)),
        ("de_f", ("crossover",)),
        ("de_variant", ("de_variant",)),
        ("gaussian_sigma", ("mutation",)),
        # Configurations without the field take the factor as their mutation probability.
        ("mutation_prob_factor", ("mutation_prob_factor", "mutation")),
        ("variation", ("crossover",)),
        ("external_archive", ("external_archive",)),
        ("selection_pressure", ("selection",)),
        ("archive_prune_policy", ("external_archive",)),
        ("blx_repair", ("repair",)),
        ("n_partitions", ("n_partitions", "pop_size")),
        ("kappa", ("kappa",)),
    ],
)
def test_space_parameters_map_to_configuration_fields(parameter: str, fields: tuple[str, ...]) -> None:
    assert config_fields_for(parameter) == fields


def test_the_smsemoa_reference_point_offset_control_changes_the_reference_point() -> None:
    form = _form("smsemoa")
    start = form.start_values()
    # The control starts at the default configuration's offset, which SMSEMOAConfig keeps inside reference_point.
    assert start["reference_point_offset"] == form.baseline.to_dict()["reference_point"]["offset"]

    resolved = form.resolve({**start, "reference_point_offset": 0.5})

    assert resolved.unapplied == ()
    assert resolved.config.to_dict()["reference_point"]["offset"] == 0.5
