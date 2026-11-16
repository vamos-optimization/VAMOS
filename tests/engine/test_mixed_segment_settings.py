"""The segment settings of a mixed tuning space reach the run, for every algorithm that has one.

A mixed space offers an operator and a probability per segment of the variables (``real_mutation``,
``int_crossover_prob``, ...). MOEA/D, NSGA-III, SMS-EMOA and SPEA2 build their mixed variation with
the shared discrete builder, which gave the operators the problem's segment layout without the
configuration's settings: every point of their mixed spaces ran as the default. Each change below must
change the configuration and the run, from the segment operators' defaults (``per_segment``, the
second choice of ``crossover`` and ``mutation`` besides the default mixed operators).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vamos.engine.tuning import config_from_assignment
from vamos.engine.tuning.grammar import builtin_parameter_space
from vamos.experiment.unified import optimize
from vamos.foundation.problem.registry import make_problem_selection

ALGORITHMS = ("moead", "nsgaiii", "smsemoa", "spea2", "nsgaii", "ibea", "agemoea", "rvea")
# mixed_design has real, integer and categorical segments.
CHANGES: dict[str, dict[str, Any]] = {
    "real_mutation_prob": {"real_mutation_prob": 0.5},
    "real_mutation": {"real_mutation": "polynomial", "real_mutation_eta": 20.0},
    "int_mutation": {"int_mutation": "creep", "int_mutation_step": 3},
    "int_crossover": {"int_crossover": "sbx", "int_crossover_eta": 20.0},
    "cat_mutation_prob": {"cat_mutation_prob": 0.5},
}


def _segment_defaults(name: str) -> dict[str, Any]:
    """The default configuration with the segment operators, each at its defaults."""
    space = builtin_parameter_space(f"{name}_mixed")
    values: dict[str, Any] = {**space.default_values(), "crossover": "per_segment", "mutation": "per_segment"}
    while True:  # the segment nodes that become active bring their default productions
        derived = {parameter.name: values.get(parameter.name, parameter.default) for parameter in space.active_parameters(values)}
        if derived == values:
            return values
        values = derived


def _final_population(name: str, change: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    values = {**_segment_defaults(name), **change}
    config = config_from_assignment(f"{name}_mixed", {**values, "pop_size": 20, "n_obj": 2})
    problem = make_problem_selection("mixed_design").instantiate()
    result = optimize(problem, algorithm=name, algorithm_config=config, max_evaluations=200, seed=3, engine="numpy")
    return config.to_dict(), np.asarray(result.X, dtype=float)


@pytest.mark.parametrize("change", sorted(CHANGES))
@pytest.mark.parametrize("name", ALGORITHMS)
def test_a_changed_segment_setting_changes_the_configuration_and_the_run(name: str, change: str) -> None:
    default_config, default_run = _final_population(name, {})
    changed_config, changed_run = _final_population(name, CHANGES[change])

    assert changed_config != default_config
    assert default_run.shape != changed_run.shape or not np.array_equal(default_run, changed_run), (
        f"{name}_mixed runs the same population with {CHANGES[change]}: the segment setting does not reach the run"
    )
