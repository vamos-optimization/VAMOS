"""Rule G7: a space's default configuration runs as its algorithm's ``default()`` configuration.

The check compares seeded runs, not configurations, because equivalent configurations can be
written differently: the space below gives polynomial mutation a probability factor of 1.0, while
``NSGAIIConfig.default()`` gives it the probability ``1/n``.
"""

from __future__ import annotations

import copy

import pytest

from vamos.engine.tuning.grammar import ParameterSpaceError, check_default_configuration, default_assignment, parameter_space_from_document
from vamos.foundation.problem.registry import make_problem_selection

from .test_parameter_space_schema import small_space


def _problem() -> object:
    return make_problem_selection("zdt1", n_var=12).instantiate()


def test_a_space_whose_defaults_are_the_algorithms_passes() -> None:
    check_default_configuration(parameter_space_from_document(small_space()), _problem(), max_evaluations=300)


@pytest.mark.parametrize(
    ("name", "default"),
    [("sbx_distribution_index", 15.0), ("crossover_probability", 1.0), ("mutation_probability_factor", 2.0), ("tournament_size", 3)],
)
def test_g7_rejects_a_default_that_runs_differently(name: str, default: object) -> None:
    document = copy.deepcopy(small_space())
    _set_default(document["parameters"], name, default)

    with pytest.raises(ParameterSpaceError) as raised:
        check_default_configuration(parameter_space_from_document(document), _problem(), max_evaluations=300)
    assert raised.value.rule == "G7"
    assert "does not run as NSGAIIConfig.default()" in str(raised.value)


def test_the_default_assignment_adds_the_run_settings() -> None:
    assignment = default_assignment(parameter_space_from_document(small_space()), pop_size=40, n_obj=2)

    assert assignment["pop_size"] == 40 and assignment["n_obj"] == 2
    assert assignment["crossover"] == "sbx"


def _set_default(nodes: dict[str, object], name: str, default: object) -> bool:
    for key, node in nodes.items():
        assert isinstance(node, dict)
        if key == name:
            node["default"] = default
            return True
        children = dict(node.get("global_sub_parameters", {}))
        values = node.get("values")
        if isinstance(values, dict):
            for production in values.values():
                children.update((production or {}).get("conditional_parameters", {}))
        if _set_default(children, name, default):
            return True
    return False
