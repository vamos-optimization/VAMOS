"""The built-in tuning spaces are YAML grammars, packaged with VAMOS, and the tuning spaces are their views.

The grammars were written once from the Python builders of VAMOS 1.3.0 with the migration M1 to M7
of the VAMOS-CONFIG design, without ``pop_size``, which is a run setting, and with ``auto`` for the
initializer and the repair. They are maintained as YAML: ``build_<space>_config_space()`` reads its
grammar.
"""

from __future__ import annotations

import inspect
from importlib import resources

import pytest
import yaml

from vamos.engine.operators.impl.real.crossover_blend import BLXAlphaBetaCrossover, BLXAlphaCrossover, WholeArithmeticCrossover
from vamos.engine.operators.impl.real.crossover_multi_parent import PCXCrossover, UNDXCrossover
from vamos.engine.operators.impl.real.crossover_stochastic import FuzzyCrossover, LaplaceCrossover
from vamos.engine.operators.impl.real.mutation import (
    CauchyMutation,
    GaussianMutation,
    LevyFlightMutation,
    LinkedPolynomialMutation,
    PolynomialMutation,
    PowerLawMutation,
    UniformMutation,
)
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
    text = (GRAMMARS / f"{name}.yaml").read_text(encoding="utf-8")
    space = parameter_space_from_yaml(text)

    assert space.name == name
    base, _, suffix = name.partition("_")
    assert space.algorithm == base
    assert space.encoding == (suffix if suffix in {"permutation", "mixed", "binary", "integer"} else "real")
    assert space.variant == name
    assert isinstance(yaml.safe_load(text), dict)  # plain YAML, which any YAML reader reads


def test_vamos_reads_the_built_in_spaces_from_their_yaml_files() -> None:
    # The YAML files are the source of truth and VAMOS reads them at run time: no generated copy ships.
    assert sorted(entry.name for entry in GRAMMARS.iterdir() if entry.is_file()) == [f"{name}.yaml" for name in SPACE_NAMES]
    for name in SPACE_NAMES:
        text = (GRAMMARS / f"{name}.yaml").read_text(encoding="utf-8")
        assert parameter_space_document(builtin_parameter_space(name)) == parameter_space_document(parameter_space_from_yaml(text))


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


REAL_OPERATOR_PARAMETERS = {
    # grammar parameter: (operator node, value, operator class, constructor argument)
    "crossover_alpha": ("crossover", "blx_alpha", BLXAlphaCrossover, "alpha"),
    "blxab_alpha": ("crossover", "blx_alpha_beta", BLXAlphaBetaCrossover, "alpha"),
    "blxab_beta": ("crossover", "blx_alpha_beta", BLXAlphaBetaCrossover, "beta"),
    "wa_alpha": ("crossover", "whole_arithmetic", WholeArithmeticCrossover, "alpha"),
    "laplace_a": ("crossover", "laplace", LaplaceCrossover, "a"),
    "laplace_b": ("crossover", "laplace", LaplaceCrossover, "b"),
    "fuzzy_d": ("crossover", "fuzzy", FuzzyCrossover, "d"),
    "pcx_sigma_eta": ("crossover", "pcx", PCXCrossover, "sigma_eta"),
    "pcx_sigma_zeta": ("crossover", "pcx", PCXCrossover, "sigma_zeta"),
    "undx_zeta": ("crossover", "undc", UNDXCrossover, "zeta"),
    "undx_eta": ("crossover", "undc", UNDXCrossover, "eta"),
    "polynomial_mutation_distribution_index": ("mutation", "polynomial", PolynomialMutation, "eta"),
    "linked_polynomial_mutation_distribution_index": ("mutation", "linked_polynomial", LinkedPolynomialMutation, "eta"),
    "gaussian_sigma": ("mutation", "gaussian", GaussianMutation, "sigma"),
    "cauchy_gamma": ("mutation", "cauchy", CauchyMutation, "gamma"),
    "uniform_perturb": ("mutation", "uniform", UniformMutation, "perturb"),
    "levy_beta": ("mutation", "levy_flight", LevyFlightMutation, "beta"),
    "levy_scale": ("mutation", "levy_flight", LevyFlightMutation, "scale"),
    "power_index": ("mutation", "power_law", PowerLawMutation, "index"),
}


@pytest.mark.parametrize("name", SPACE_NAMES)
def test_the_real_operators_parameters_default_to_their_constructor_values(name: str) -> None:
    # M4: an operator parameter takes the operator's own default, which the bridge also falls back to.
    # SBX is the exception: every configuration sets its index (20, 30 or 15), the class's 10 never runs.
    space = grammar(name)
    if space.encoding != "real":
        return
    for parameter, (node, value, operator, argument) in REAL_OPERATOR_PARAMETERS.items():
        if parameter not in space.names():
            continue
        if node in space.names():  # SMPSO's polynomial mutation is fixed, with its index at the top
            assert parameter in {child.name for child in space.parameter(node).production(value)}, parameter
        own = inspect.signature(operator).parameters[argument].default
        assert space.parameter(parameter).default == own, (parameter, own)


# The constraint-handling techniques have no default in VAMOS, jMetal or Evolver, so the grammars take the
# settings of the papers that introduced them.
PUBLISHED_TECHNIQUE_SETTINGS = {
    # Takahama and Sakai's epsilon level control: epsilon(0) the violation of the individual at 0.2N, cp = 5
    # and Tc = 0.8 Tmax (S. Sakai, T. Takahama, RIMS Kokyuroku 1584:90-101, 2008, eq. (10)).
    "epsilon_initial_quantile": 0.2,
    "epsilon_exponent": 5.0,
    "epsilon_decay_fraction": 0.8,
    # Fan et al.'s push and pull search: threshold 1e-3 on the change of the ideal and nadir points over
    # l = 20 of 1000 generations, Tc = 0.8 Tmax, cp = 2, alpha = 0.95 and tau = 0.1 (arXiv:1709.05915,
    # eq. (4) and section IV-A).
    "push_pull_change_threshold": 1e-3,
    "push_pull_window_fraction": 0.02,
    "push_pull_decay_fraction": 0.8,
    "push_pull_exponent": 2.0,
    "push_pull_feasibility_threshold": 0.95,
    "push_pull_reduction_rate": 0.1,
    # Runarsson and Yao's P_f (IEEE TEC 4(3):284-294, 2000, table II).
    "stochastic_ranking_probability": 0.45,
    # Fan et al.'s improved epsilon in MOEA/D, as MOEA/D-IEpsilon in the push and pull paper: tau = 0.1,
    # alpha = 0.95 and Tc = 0.8 Tmax (arXiv:1709.05915, section IV-A).
    "improved_epsilon_tao": 0.1,
    "improved_epsilon_feasibility_ratio_threshold": 0.95,
    "improved_epsilon_decay_fraction": 0.8,
    # Development after 1.5.2. Joines and Houck's non-stationary penalty, (C * k)^alpha * SVC(beta, X): their
    # P(2, 2), with C = 0.5 (IEEE ICEC 1994, pp. 579-584, sections 3.1 and 4).
    "dynamic_penalty_constant": 0.5,
    "dynamic_penalty_alpha": 2.0,
    "dynamic_penalty_beta": 2.0,
    # A static penalty has no published setting; the grammars take pymoo's default penalty.
    "static_penalty_coefficient": 1e6,
}


def test_every_published_setting_has_a_parameter() -> None:
    names = {name for space in SPACE_NAMES for name in grammar(space).names()}
    assert set(PUBLISHED_TECHNIQUE_SETTINGS) <= names


@pytest.mark.parametrize("name", SPACE_NAMES)
def test_the_constraint_handling_parameters_default_to_the_published_settings(name: str) -> None:
    space = grammar(name)
    for parameter, published in PUBLISHED_TECHNIQUE_SETTINGS.items():
        if parameter in space.names():
            assert space.parameter(parameter).default == pytest.approx(published), parameter


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
    # the algorithms other than NSGA-II, and the segments of the mixed operators. The default mixed
    # mutation takes the factor: its probability is 1/n.
    space = grammar(name)
    names = set(space.names())
    if space.encoding == "permutation" and space.algorithm != "nsgaii":
        assert "mutation_probability_factor" not in names
        return
    if space.encoding == "mixed":
        assert [child.name for child in space.parameter("mutation").production("mixed")] == ["mutation_probability_factor"]
    assert "mutation_probability_factor" in names and "mutation_prob" not in names
    factor = space.parameter("mutation_probability_factor")
    assert factor.range == (0.25, 3.0)
    assert factor.default == (2.0 if name == "nsgaii_permutation" else 1.0)
