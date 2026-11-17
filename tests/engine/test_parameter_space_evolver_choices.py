"""MOEA/D and SMS-EMOA offer the operators and initializers that Evolver gives them and VAMOS runs.

A configuration tuned or published with Evolver for these two algorithms should be expressible in
VAMOS, so their real-coded grammars (and the constrained variants built on them) offer the Sobol,
opposition-based and Halton initializers of NSGA-II's space (Evolver's Cauchy initialization has no
VAMOS counterpart), the Levy flight and power-law mutations, and the two-parent crossovers of NSGA-II's
space: Laplace and fuzzy recombination in both, and, for MOEA/D, BLX-alpha, BLX-alpha-beta, whole
arithmetic and arithmetic, under a ``variation`` node as in Evolver's ``MOEADDouble``. PCX and UNDC
stay out: MOEA/D and SMS-EMOA mate two parents. The SBX and polynomial indices span Evolver's
[5, 400].
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from vamos.engine.tuning import config_from_assignment
from vamos.engine.tuning.grammar import ParameterSpace, builtin_parameter_space, configuration
from vamos.experiment.unified import optimize
from vamos.foundation.problem.registry import make_problem_selection

from .test_builtin_parameter_space_defaults import PROBLEMS, _problem_kind

TWO_PARENT_CROSSOVERS = ("blx_alpha", "blx_alpha_beta", "whole_arithmetic", "arithmetic", "laplace", "fuzzy")
NEW_CHOICES: dict[str, dict[str, tuple[str, ...]]] = {
    "moead": {
        "initial_solutions": ("sobol", "obl", "halton"),
        "mutation": ("levy_flight", "power_law"),
        "variation": ("crossover_and_mutation",),
        "crossover": ("sbx", *TWO_PARENT_CROSSOVERS),
    },
    "smsemoa": {
        "initial_solutions": ("sobol", "obl", "halton"),
        "mutation": ("levy_flight", "power_law"),
        "crossover": ("blx_alpha_beta", "whole_arithmetic", "laplace", "fuzzy"),
    },
}
SPACES = ("moead", "moead_constrained", "smsemoa", "smsemoa_constrained")
CASES = [
    pytest.param(name, node, value, id=f"{name}-{node}-{value}")
    for name in SPACES
    for node, values in NEW_CHOICES[name.split("_")[0]].items()
    for value in values
]
WIDE_INDICES = ("sbx_distribution_index", "polynomial_mutation_distribution_index", "linked_polynomial_mutation_distribution_index")


def _route(space: ParameterSpace, target: str) -> dict[str, object]:
    """The choices of the categorical nodes above ``target`` that make it active."""

    def visit(parameters: tuple, chosen: dict[str, object]) -> dict[str, object] | None:
        for parameter in parameters:
            if parameter.name == target:
                return chosen
            if parameter.type == "categorical":
                for value in parameter.values:
                    found = visit(parameter.children(value), {**chosen, parameter.name: value})
                    if found is not None:
                        return found
        return None

    route = visit(space.parameters, {})
    assert route is not None, f"{space.name} has no parameter {target!r}"
    return route


def _with_choice(space: ParameterSpace, node: str, value: object) -> dict[str, object]:
    """The default configuration with ``node`` at ``value``: its route chosen and every new parameter at its default.

    ``configuration`` rejects a value that the grammar does not offer.
    """
    chosen: Mapping[str, object] = {**_route(space, node), node: value}
    values = {**space.default_values(), **chosen}
    while True:  # a node that becomes active brings its default production
        derived = {parameter.name: values.get(parameter.name, parameter.default) for parameter in space.active_parameters(values)}
        if derived == values:
            return configuration(space, derived).to_values()
        values = derived


@pytest.mark.parametrize(("name", "node", "value"), CASES)
def test_the_grammar_offers_the_choice(name: str, node: str, value: str) -> None:
    space = builtin_parameter_space(name)
    assert node in space.names()
    assert value in space.parameter(node).values


@pytest.mark.parametrize(("name", "node", "value"), CASES)
def test_the_choice_builds_and_runs(name: str, node: str, value: str) -> None:
    space = builtin_parameter_space(name)
    assignment = _with_choice(space, node, value)
    key, kwargs = PROBLEMS[_problem_kind(name)]
    problem = make_problem_selection(key, **kwargs).instantiate()

    config = config_from_assignment(name, {**assignment, "pop_size": 12, "n_obj": int(problem.n_obj)})
    built = config.to_dict()
    if node == "initial_solutions":
        assert built["initializer"]["type"] == value
    elif node == "mutation":
        assert built["mutation"][0] == value
    elif node in {"crossover", "variation"}:
        assert built["crossover"][0] == assignment.get("crossover", "de")
    result = optimize(problem, algorithm=space.algorithm, algorithm_config=config, max_evaluations=48, seed=1, engine="numpy")
    assert np.isfinite(np.asarray(result.F, dtype=float)).all()


@pytest.mark.parametrize("name", SPACES)
def test_the_indices_span_evolvers_range(name: str) -> None:
    space = builtin_parameter_space(name)
    for index in WIDE_INDICES:
        assert space.parameter(index).range == (5.0, 400.0), index
        assert space.parameter(index).default == 20.0, index


def test_opposition_based_initialization_has_evolvers_name() -> None:
    from vamos.engine.tuning.racing.parameter_names import from_evolver, to_evolver

    assert to_evolver({"initial_solutions": "obl"}) == {"createInitialSolutions": "oppositionBased"}
    assert from_evolver({"createInitialSolutions": "oppositionBased"}) == {"initial_solutions": "obl"}
