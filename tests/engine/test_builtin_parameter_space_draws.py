"""Every configuration that a built-in grammar can draw builds a valid algorithm configuration.

The schema has no forbidden clauses, so a grammar must offer only configurations that run: each test
draws many seeded configurations of every space, from the grammar and from its tuning view (the
``AlgorithmConfigSpace`` that the tuners sample), and builds each one. A few of them also run.
"""

from __future__ import annotations

import numpy as np
import pytest

from vamos.engine.tuning import config_from_assignment
from vamos.engine.tuning.grammar import builtin_parameter_space, builtin_parameter_space_names, configuration
from vamos.engine.tuning.racing import bridge_spaces
from vamos.experiment.unified import optimize
from vamos.foundation.problem.registry import make_problem_selection

from .test_builtin_parameter_space_defaults import PROBLEMS, _problem_kind

SPACE_NAMES = sorted(builtin_parameter_space_names())


@pytest.mark.parametrize("name", SPACE_NAMES)
def test_every_grammar_draw_builds_a_configuration(name: str) -> None:
    space = builtin_parameter_space(name)
    rng = np.random.default_rng(20261116)
    for _ in range(60):
        draw = space.sample(rng)
        assert configuration(space, draw).to_values() == draw
        for n_obj in (2, 3):
            config_from_assignment(name, {**draw, "pop_size": 60, "n_obj": n_obj})


@pytest.mark.parametrize("name", SPACE_NAMES)
def test_every_draw_of_the_tuning_view_is_a_configuration_of_the_grammar(name: str) -> None:
    space = builtin_parameter_space(name)
    view = getattr(bridge_spaces, f"build_{name}_config_space")()
    param_space = view.to_param_space()
    rng = np.random.default_rng(7)
    for _ in range(60):
        # Both samplers of the view: the dependency-ordered one, and the declaration-order one of the racing tuner.
        for draw in (view.sample(rng), param_space.sample(rng)):
            assert configuration(space, draw).to_values() == {key: draw[key] for key in configuration(space, draw).to_values()}
            assert set(configuration(space, draw).to_values()) == set(draw)


@pytest.mark.parametrize("name", SPACE_NAMES)
def test_grammar_draws_run(name: str) -> None:
    space = builtin_parameter_space(name)
    key, kwargs = PROBLEMS[_problem_kind(name)]
    problem = make_problem_selection(key, **kwargs).instantiate()
    rng = np.random.default_rng(3)
    for seed in range(3):
        draw = space.sample(rng)
        config = config_from_assignment(name, {**draw, "pop_size": 12, "n_obj": int(problem.n_obj)})
        result = optimize(problem, algorithm=space.algorithm, algorithm_config=config, max_evaluations=60, seed=seed, engine="numpy")
        assert np.isfinite(np.asarray(result.F, dtype=float)).all(), draw
