"""The tuning backends read the built-in spaces' grammars, not condition expressions.

A built-in space reaches every tuner as the flat ``ParamSpace`` view of its grammar, which keeps the
grammar. ConfigSpace (SMAC3 and BOHB) gets hyperparameters with the grammar's defaults and its tree as
conditions; Optuna suggests the active parameters depth-first and starts from the default
configuration; irace gets one condition per tree edge and the default configuration as its first
initial configuration; racing samples by walking the tree and races the default configuration with
the draws.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vamos.engine.tuning import Instance, ModelBasedTuner, RacingTuner, Scenario, TuningTask
from vamos.engine.tuning._irace_space import configurations_file, parameter_file
from vamos.engine.tuning.grammar import Parameter, ParameterSpace, builtin_parameter_space, builtin_parameter_space_names, configuration
from vamos.engine.tuning.grammar.view import to_param_space

SPACE_NAMES = sorted(builtin_parameter_space_names())


def _plain(values: dict[str, Any]) -> dict[str, Any]:
    """Python scalars for the numpy scalars that ConfigSpace returns."""
    return {name: value.item() if isinstance(value, np.generic) else value for name, value in values.items()}


def _edges(space: ParameterSpace) -> list[tuple[Parameter, Parameter | None, object]]:
    """(child, parent, value) for every parameter: value None for a global sub-parameter, parent None at the root."""
    edges: list[tuple[Parameter, Parameter | None, object]] = [(parameter, None, None) for parameter in space.parameters]
    for parent in space.walk():
        edges += [(child, parent, None) for child in parent.global_sub_parameters]
        edges += [(child, parent, value) for value, production in zip(parent.values, parent.productions) for child in production]
    return edges


def _task(name: str) -> TuningTask:
    return TuningTask(
        name=f"grammar_{name}",
        param_space=to_param_space(builtin_parameter_space(name)),
        instances=[Instance(name="zdt1", n_var=6)],
        seeds=[0],
        budget_per_run=10,
        maximize=True,
    )


def test_the_view_of_a_grammar_keeps_it_and_samples_by_walking_the_tree() -> None:
    space = builtin_parameter_space("nsgaii")
    view = to_param_space(space)

    assert view.grammar is space
    for seed in range(20):
        assert view.sample(np.random.default_rng(seed)) == space.sample(np.random.default_rng(seed))


def test_without_removes_a_parameter_and_its_subtree() -> None:
    space = builtin_parameter_space("nsgaii")
    pruned = space.without({"external_archive"})

    assert {"external_archive", "archive_prune_policy"} & set(pruned.names()) == set()
    assert set(pruned.names()) == set(space.names()) - {"external_archive", "archive_prune_policy"}
    assert pruned.identity.sha256 != space.identity.sha256
    with pytest.raises(KeyError):
        space.without({"no_such_parameter"})


def test_without_values_drops_choices_and_their_productions() -> None:
    space = builtin_parameter_space("smsemoa")
    pruned = space.without_values("initial_solutions", {"obl", "scatter"})

    assert pruned.parameter("initial_solutions").values == tuple(
        value for value in space.parameter("initial_solutions").values if value not in {"obl", "scatter"}
    )
    assert "scatter_base_size_factor" not in pruned.names()
    assert pruned.identity.sha256 != space.identity.sha256 and pruned.default_values() == space.default_values()
    with pytest.raises(ValueError, match="keeps its default"):
        space.without_values("initial_solutions", {"auto"})


@pytest.mark.parametrize("name", SPACE_NAMES)
def test_configspace_carries_the_grammar_defaults_and_its_tree(name: str) -> None:
    pytest.importorskip("ConfigSpace")
    from ConfigSpace.conditions import EqualsCondition, InCondition

    from vamos.engine.tuning._model_backend_utils import build_configspace

    space = builtin_parameter_space(name)
    cs = build_configspace(to_param_space(space), seed=1)
    conditions = {condition.child.name: condition for condition in cs.conditions}

    default = _plain(dict(cs.get_default_configuration()))
    expected = space.default_values()
    assert set(default) == set(expected)
    for key, value in expected.items():
        # ConfigSpace keeps a float default to about 13 digits (ToP's 1/3).
        assert default[key] == pytest.approx(value, rel=1e-12) if isinstance(value, float) else default[key] == value, key
    for child, parent, value in _edges(space):
        if parent is None or (value is None and parent in space.parameters):
            assert child.name not in conditions, child.name
        elif value is None:
            condition = conditions[child.name]
            assert isinstance(condition, InCondition) and condition.parent.name == parent.name
            assert list(condition.values) == list(parent.values)
        else:
            condition = conditions[child.name]
            assert isinstance(condition, EqualsCondition) and (condition.parent.name, condition.value) == (parent.name, value)
    for sampled in cs.sample_configuration(20):
        configuration(space, _plain(dict(sampled)))


@pytest.mark.parametrize("backend", ["optuna", "bohb_optuna"])
def test_optuna_starts_from_the_default_configuration(backend: str) -> None:
    pytest.importorskip("optuna")
    space = builtin_parameter_space("ccmo")
    seen: list[dict[str, Any]] = []

    def evaluate(config: dict[str, Any], _ctx: Any) -> float:
        seen.append(dict(config))
        return float(len(seen))

    ModelBasedTuner(task=_task("ccmo"), max_trials=6, backend=backend, seed=2).run(evaluate)

    assert seen[0] == space.default_values()
    for config in seen:
        assert configuration(space, config).to_values() == config


def test_irace_parameters_follow_the_tree() -> None:
    space = builtin_parameter_space("ccmo")
    lines = {line.split()[0]: line for line in parameter_file(to_param_space(space)).splitlines()}

    # A production's parameter depends on its parent's value, a global sub-parameter of a node below
    # the root on every value of that node, a root parameter on nothing.
    assert lines["sbx_distribution_index"].endswith('| crossover == "sbx"')
    assert lines["crossover"].endswith('| variation == "crossover_and_mutation"')
    assert lines["crossover_probability"].endswith('| crossover %in% c("sbx", "blx_alpha", "arithmetic")')
    assert "|" not in lines["variation"] and "|" not in lines["mutation_probability_factor"]


def test_irace_starts_from_the_default_configuration() -> None:
    from vamos.engine.tuning._model_backend_irace import initial_configurations

    space = builtin_parameter_space("nsgaii")
    tuner = ModelBasedTuner(task=_task("nsgaii"), max_trials=4, backend="irace")
    given = {**space.default_values(), "crossover": "blx_alpha", "crossover_alpha": 0.3}
    given.pop("sbx_distribution_index")
    tuner_with_one = ModelBasedTuner(task=_task("nsgaii"), max_trials=4, backend="irace", irace_initial_configs=[given])

    assert initial_configurations(tuner) == [space.default_values()]
    assert initial_configurations(tuner_with_one) == [space.default_values(), given]
    configurations_file(to_param_space(space), initial_configurations(tuner_with_one))


def test_racing_races_the_default_configuration_with_its_draws() -> None:
    space = builtin_parameter_space("nsgaii")
    tuner = RacingTuner(task=_task("nsgaii"), scenario=Scenario(max_experiments=10), seed=4, max_initial_configs=5)
    configs = [state.config for state in tuner._sample_initial_configs()]

    assert configs[0] == space.default_values() and len(configs) == 5
    for config in configs:
        configuration(space, config)
