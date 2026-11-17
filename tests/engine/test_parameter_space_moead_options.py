"""The MOEA/D grammars offer the options of Evolver's MOEA/D that VAMOS runs (VAMOS-Studio's C2, C4 and C5).

Every MOEA/D space chooses the order of the subproblems, the aggregations ``augmented_tchebycheff``,
``inverted_pbi`` and ``inverse_weight_tchebycheff``, and the normalization of the objectives, which
MOEA/D runs in every encoding. The real-coded spaces also choose the DE scheme and the base of the
mutant under ``variation: differential_evolution``. The schema has no forbidden clauses, so no
derivation may ask for more random DE parents than the neighbourhood holds: the neighbourhood's
range starts at the largest number that any scheme and base need.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from vamos.engine.operators.policies.moead import DE_BASES, DE_SCHEMES, de_random_parents
from vamos.engine.tuning import config_from_assignment
from vamos.engine.tuning.grammar import ParameterSpace, builtin_parameter_space, configuration
from vamos.engine.tuning.racing.parameter_names import canonical_assignment, from_evolver, to_evolver
from vamos.experiment.unified import optimize
from vamos.foundation.problem.registry import make_problem_selection

from .test_builtin_parameter_space_defaults import PROBLEMS, _problem_kind

MOEAD_SPACES = ("moead", "moead_constrained", "moead_binary", "moead_integer", "moead_permutation", "moead_mixed")
DE_SPACES = ("moead", "moead_constrained")
AGGREGATIONS = ("augmented_tchebycheff", "inverted_pbi", "inverse_weight_tchebycheff")
# (node, value) of each new choice
OPTIONS = [
    ("subproblem_order", "cyclic"),
    *(("aggregation_function", value) for value in AGGREGATIONS),
    ("normalize_objectives", "true"),
]
DE_OPTIONS = [*(("de_scheme", scheme) for scheme in DE_SCHEMES), ("de_base", "random")]
CASES = [pytest.param(name, node, value, id=f"{name}-{node}-{value}") for name in MOEAD_SPACES for node, value in OPTIONS] + [
    pytest.param(name, node, value, id=f"{name}-{node}-{value}") for name in DE_SPACES for node, value in DE_OPTIONS
]


def _route(space: ParameterSpace, target: str) -> dict[str, object]:
    """The choices of the categorical nodes above ``target`` that make it active."""

    def visit(parameters: tuple, chosen: dict[str, object]) -> dict[str, object] | None:
        for parameter in parameters:
            if parameter.name == target:
                return chosen
            if parameter.type == "categorical":
                for found in (visit(parameter.global_sub_parameters, chosen),):
                    if found is not None:
                        return found
                for value in parameter.values:
                    found = visit(parameter.children(value), {**chosen, parameter.name: value})
                    if found is not None:
                        return found
        return None

    route = visit(space.parameters, {})
    assert route is not None, f"{space.name} has no parameter {target!r}"
    return route


def _with(space: ParameterSpace, choices: Mapping[str, object]) -> dict[str, object]:
    """The default configuration with ``choices`` (and their routes) and every new parameter at its default."""
    values: dict[str, object] = {**space.default_values()}
    for node, value in choices.items():
        values.update({**_route(space, node), node: value})
    while True:  # a node that becomes active brings its default production
        derived = {parameter.name: values.get(parameter.name, parameter.default) for parameter in space.active_parameters(values)}
        if derived == values:
            return configuration(space, derived).to_values()
        values = derived


def _run(name: str, assignment: Mapping[str, object]) -> dict[str, object]:
    space = builtin_parameter_space(name)
    key, kwargs = PROBLEMS[_problem_kind(name)]
    problem = make_problem_selection(key, **kwargs).instantiate()
    config = config_from_assignment(name, {**assignment, "pop_size": 12, "n_obj": int(problem.n_obj)})
    result = optimize(problem, algorithm=space.algorithm, algorithm_config=config, max_evaluations=48, seed=1, engine="numpy")
    assert np.isfinite(np.asarray(result.F, dtype=float)).all()
    return config.to_dict()


@pytest.mark.parametrize("name", MOEAD_SPACES)
def test_every_moead_space_offers_the_order_the_aggregations_and_the_normalization(name: str) -> None:
    space = builtin_parameter_space(name)
    order = space.parameter("subproblem_order")
    assert (order.values, order.default) == (("random_permutation", "cyclic"), "random_permutation")
    aggregation = space.parameter("aggregation_function")
    assert set(AGGREGATIONS) <= set(aggregation.values)
    assert [child.name for child in aggregation.production("augmented_tchebycheff")] == ["atch_rho"]
    assert [child.name for child in aggregation.production("inverted_pbi")] == ["ipbi_theta"]
    assert space.parameter("atch_rho").default == 1e-4 and space.parameter("ipbi_theta").default == 0.1
    normalize = space.parameter("normalize_objectives")
    assert normalize.name in {child.name for child in aggregation.global_sub_parameters}
    assert (normalize.values, normalize.default) == (("false", "true"), "false")
    assert [child.name for child in normalize.production("true")] == ["normalization_epsilon"]
    assert space.parameter("normalization_epsilon").default == 1e-6


@pytest.mark.parametrize("name", DE_SPACES)
def test_the_de_variation_chooses_its_scheme_and_base(name: str) -> None:
    space = builtin_parameter_space(name)
    children = {child.name for child in space.parameter("variation").production("differential_evolution")}
    assert {"de_scheme", "de_base"} <= children
    assert (space.parameter("de_scheme").values, space.parameter("de_scheme").default) == (DE_SCHEMES, "rand_1_bin")
    assert (space.parameter("de_base").values, space.parameter("de_base").default) == (DE_BASES, "current")


@pytest.mark.parametrize(("name", "node", "value"), CASES)
def test_each_option_builds_and_runs(name: str, node: str, value: str) -> None:
    assignment = _with(builtin_parameter_space(name), {node: value})
    built = _run(name, assignment)

    if node == "subproblem_order":
        assert built["subproblem_order"] == value
    elif node == "aggregation_function":
        assert built["aggregation"][0] == value
    elif node == "normalize_objectives":
        assert built["aggregation"][1]["normalize"] is True and built["aggregation"][1]["epsilon"] == 1e-6
    else:
        # The bridge leaves out the defaults (rand_1_bin, current), so their configuration is the one of VAMOS 1.3.0.
        key, default = {"de_scheme": ("scheme", "rand_1_bin"), "de_base": ("base", "current")}[node]
        assert built["crossover"][0] == "de" and built["crossover"][1].get(key, default) == value


@pytest.mark.parametrize("name", DE_SPACES)
def test_no_derivation_asks_for_more_de_parents_than_the_neighbourhood_holds(name: str) -> None:
    # rand/2 with a random base needs five random parents; the neighbourhood's range starts there.
    space = builtin_parameter_space(name)
    low = space.parameter("neighborhood_size").range[0]
    assert low >= max(de_random_parents(scheme, base) for scheme in DE_SCHEMES for base in DE_BASES)
    for scheme in DE_SCHEMES:
        for base in DE_BASES:
            _run(name, _with(space, {"de_scheme": scheme, "de_base": base, "neighborhood_size": int(low)}))
    rng = np.random.default_rng(337)
    for _ in range(200):
        config_from_assignment(name, {**space.sample(rng), "pop_size": 60, "n_obj": 2})


@pytest.mark.parametrize("name", MOEAD_SPACES)
def test_the_defaults_build_the_configuration_of_vamos_1_3_0(name: str) -> None:
    # The new options are off by default, and the bridge leaves them out: same configuration, same task id.
    space = builtin_parameter_space(name)
    built = config_from_assignment(name, {**space.default_values(), "pop_size": 100, "n_obj": 2}).to_dict()
    assert "subproblem_order" not in built
    assert set(built["aggregation"][1]) <= {"theta", "rho"}
    if name in DE_SPACES:
        assert set(built["crossover"][1]) == {"cr", "f"}


def test_the_new_options_have_evolvers_names() -> None:
    assignment = {
        "subproblem_order": "cyclic",
        "aggregation_function": "augmented_tchebycheff",
        "atch_rho": 1e-4,
        "normalize_objectives": "true",
        "normalization_epsilon": 1e-3,
        "variation": "differential_evolution",
        "de_scheme": "rand_2_exp",
        "de_base": "random",
    }
    evolver = to_evolver(assignment)

    assert evolver["subProblemIdGenerator"] == "cyclicIntegerSequence"
    assert evolver["aggregationFunction"] == "augmentedTschebyscheff"
    assert (evolver["normalizeObjectives"], evolver["epsilonParameterForNormalization"]) == ("true", 1e-3)
    assert (evolver["variation"], evolver["differentialEvolutionCrossover"]) == ("differentialEvolutionVariation", "RAND_2_EXP")
    assert "de_base" not in evolver  # Evolver's MOEA/D always draws the base at random
    assert from_evolver(evolver) == canonical_assignment(assignment)


def test_evolvers_inverted_pbi_always_normalizes() -> None:
    read = from_evolver({"aggregationFunction": "invertedPenaltyBoundaryIntersection", "ipbiTheta": 0.5, "normalizeObjectives": "false"})

    assert read == {
        "aggregation_function": "inverted_pbi",
        "ipbi_theta": 0.5,
        "normalize_objectives": "true",
        "normalization_epsilon": 1e-6,  # jMetal's default epsilon, which Evolver keeps without normalizeObjectives
    }
    written = to_evolver({**read, "normalization_epsilon": 1e-4})
    assert (written["aggregationFunction"], written["normalizeObjectives"]) == ("invertedPenaltyBoundaryIntersection", "true")
    with pytest.raises(ValueError, match="always normalizes"):
        to_evolver({"aggregation_function": "inverted_pbi", "ipbi_theta": 0.5, "normalize_objectives": "false"})


def test_evolvers_de_variation_draws_a_random_base() -> None:
    read = from_evolver(
        {"variation": "differentialEvolutionVariation", "differentialEvolutionCrossover": "RAND_1_BIN", "CR": 1.0, "F": 0.5}
    )

    assert read["de_base"] == "random" and read["de_scheme"] == "rand_1_bin"
    with pytest.raises(ValueError, match="draws the base of the DE mutant at random"):
        to_evolver({"variation": "differential_evolution", "de_scheme": "rand_1_bin", "de_base": "current"})


@pytest.mark.parametrize("name", DE_SPACES)
def test_a_moead_configuration_round_trips_through_evolvers_command_line(name: str) -> None:
    from vamos.engine.tuning.grammar import configuration_from_command_line

    space = builtin_parameter_space(name)
    values = _with(space, {"de_base": "random", "aggregation_function": "inverted_pbi", "normalize_objectives": "true"})
    built = configuration(space, values)

    assert configuration_from_command_line(space, built.to_command_line(evolver=True), evolver=True) == built
