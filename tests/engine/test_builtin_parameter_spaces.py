"""The built-in tuning spaces are YAML grammars, packaged with VAMOS, and the tuning spaces are their views.

The grammars were written once from the Python builders of VAMOS 1.3.0 with the migration M1 to M7
of the VAMOS-CONFIG design, without ``pop_size``, which is a run setting, and with ``auto`` for the
initializer and the repair. They are maintained as YAML: ``build_<space>_config_space()`` reads its
grammar.
"""

from __future__ import annotations

import json
from importlib import resources

import pytest

from vamos.engine.tuning.grammar import (
    ParameterSpace,
    builtin_parameter_space,
    builtin_parameter_space_names,
    parameter_space_document,
    parameter_space_from_yaml,
)
from vamos.engine.tuning.racing import bridge_spaces
from vamos.engine.tuning.racing.param_space import Boolean, Categorical, Int, Real
from vamos.engine.tuning.racing.parameter_names import PARAMETER_NAMES

SPACE_NAMES = sorted(name[len("build_") : -len("_config_space")] for name in bridge_spaces.__all__)
GRAMMARS = resources.files("vamos.resources") / "parameter_spaces"


def grammar(name: str) -> ParameterSpace:
    return builtin_parameter_space(name)


def test_every_built_in_space_has_a_grammar() -> None:
    assert sorted(builtin_parameter_space_names()) == SPACE_NAMES
    assert sorted(entry.name for entry in GRAMMARS.iterdir() if entry.name.endswith(".yaml")) == [f"{name}.yaml" for name in SPACE_NAMES]


@pytest.mark.parametrize("name", SPACE_NAMES)
def test_every_grammar_validates_and_names_its_algorithm_and_encoding(name: str) -> None:
    yaml = pytest.importorskip("yaml")
    text = (GRAMMARS / f"{name}.yaml").read_text(encoding="utf-8")
    space = parameter_space_from_yaml(text)

    assert space.name == name
    base, _, suffix = name.partition("_")
    assert space.algorithm == base
    assert space.encoding == (suffix if suffix in {"permutation", "mixed", "binary", "integer"} else "real")
    assert space.variant == name
    assert isinstance(yaml.safe_load(text), dict)


def test_the_packaged_bundle_is_the_yaml_files() -> None:
    pytest.importorskip("yaml")
    bundle = json.loads((GRAMMARS / "parameter_spaces.json").read_text(encoding="utf-8"))
    from_yaml = {
        name: parameter_space_document(parameter_space_from_yaml((GRAMMARS / f"{name}.yaml").read_text(encoding="utf-8")))
        for name in SPACE_NAMES
    }

    assert bundle == {"schema": "vamos.parameter-space", "schema_version": "1.0.0", "spaces": from_yaml}, (
        "The YAML grammars and parameter_spaces.json differ: run python tools/update_parameter_spaces.py"
    )


@pytest.mark.parametrize("name", SPACE_NAMES)
def test_the_tuning_space_is_the_view_of_its_grammar(name: str) -> None:
    space = grammar(name)
    view = getattr(bridge_spaces, f"build_{name}_config_space")()
    params = view.to_param_space().params

    assert view.algorithm_name == name
    assert sorted(params) == sorted(space.names())
    for parameter in space.walk():
        spec = params[parameter.name]
        if parameter.type == "categorical":
            assert isinstance(spec, Categorical) and list(spec.choices) == list(parameter.values)
        elif parameter.type == "boolean":
            assert isinstance(spec, Boolean)
        else:
            assert isinstance(spec, Int if parameter.type == "int" else Real)
            assert (spec.low, spec.high, spec.log) == (*parameter.range, parameter.log)  # type: ignore[misc]
        assert spec.role == parameter.role


@pytest.mark.parametrize("name", SPACE_NAMES)
def test_pop_size_is_a_run_setting_not_a_tuned_parameter(name: str) -> None:
    view = getattr(bridge_spaces, f"build_{name}_config_space")()

    assert "pop_size" not in grammar(name).names()
    assert "pop_size" not in view.to_param_space().params
    assert "n_partitions" not in grammar(name).names()


@pytest.mark.parametrize("name", SPACE_NAMES)
def test_every_default_is_inside_its_domain_and_auto_is_the_default_initializer_and_repair(name: str) -> None:
    space = grammar(name)
    for parameter in space.walk():
        assert parameter.contains(parameter.default), parameter.name
    for choice in ("initial_solutions", "repair"):
        if choice in space.names():
            assert space.parameter(choice).values[0] == "auto"
            assert space.parameter(choice).default == "auto"
    if space.encoding == "real" and space.algorithm not in {"smpso"}:
        assert {"initial_solutions", "repair"} <= set(space.names())


@pytest.mark.parametrize("name", SPACE_NAMES)
def test_the_external_archive_is_one_choice(name: str) -> None:
    # M1: none, unbounded or bounded, with the pruning policy under bounded, as Evolver's algorithmResult and archiveType.
    space = grammar(name)
    assert not {"use_external_archive", "archive_unbounded"} & set(space.names())
    if space.algorithm in {"ccmo", "ctaea", "top", "cmoeams"}:
        assert "external_archive" not in space.names()
        return
    archive = space.parameter("external_archive")
    assert archive.values == ("none", "unbounded", "bounded") and archive.default == "none"
    assert [child.name for child in archive.production("bounded")] == ["archive_prune_policy"]


@pytest.mark.parametrize("name", SPACE_NAMES)
def test_each_polynomial_mutation_has_its_own_distribution_index(name: str) -> None:
    # M2: one parent per parameter, so the index of the two polynomial mutations is two parameters.
    space = grammar(name)
    if "mutation" not in space.names() or space.encoding != "real":
        return
    mutation = space.parameter("mutation")
    assert [child.name for child in mutation.production("polynomial")] == ["polynomial_mutation_distribution_index"]
    assert [child.name for child in mutation.production("linked_polynomial")] == ["linked_polynomial_mutation_distribution_index"]


@pytest.mark.parametrize("name", SPACE_NAMES)
def test_operator_rates_hang_from_their_operator(name: str) -> None:
    # M5: a subtree crossover of two configurations moves an operator with its rate.
    space = grammar(name)
    roots = {parameter.name for parameter in space.parameters}
    assert not {"crossover_probability", "mutation_probability_factor", "mutation_prob"} & roots - (
        {"mutation_probability_factor"} if name.startswith("smpso") else set()
    )
    for operator, rate in (
        ("crossover", "crossover_probability"),
        ("mutation", "mutation_probability_factor"),
        ("mutation", "mutation_prob"),
    ):
        if rate in space.names() and operator in space.names():
            node = space.parameter(operator)
            assert rate in {child.name for child in node.global_sub_parameters} | {
                child.name for production in node.productions for child in production
            }, (operator, rate)


@pytest.mark.parametrize("name", SPACE_NAMES)
def test_one_name_per_operator_and_parameter(name: str) -> None:
    # M6: polynomial mutation is "polynomial" in every space, and the parameters have their canonical names.
    space = grammar(name)
    earlier = {vamos for vamos, canonical, _ in PARAMETER_NAMES if vamos != canonical}
    assert not earlier & set(space.names())
    values = {value for parameter in space.walk() for value in parameter.values}
    assert not {"pm", "undx"} & values
    # No value takes another operator's name: the real-coded reset keeps "uniform_reset" ("random_reset" is the integer one).
    if space.encoding == "real" and "mutation" in space.names():
        assert "random_reset" not in space.parameter("mutation").values


@pytest.mark.parametrize("name", SPACE_NAMES)
def test_the_mutation_probability_is_a_factor_of_one_over_n(name: str) -> None:
    # M7, except where the default is a constant probability per solution: the permutation mutations of
    # the algorithms other than NSGA-II, and the segments of the mixed operators.
    space = grammar(name)
    names = set(space.names())
    if space.encoding == "mixed" or (space.encoding == "permutation" and space.algorithm != "nsgaii"):
        assert "mutation_probability_factor" not in names
        return
    assert "mutation_probability_factor" in names and "mutation_prob" not in names
    factor = space.parameter("mutation_probability_factor")
    assert factor.range == (0.25, 3.0)
    assert factor.default == (2.0 if name == "nsgaii_permutation" else 1.0)
