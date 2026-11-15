"""SMS-EMOA's optional reference point of jMetal and Evolver.

jMetal 7.11's ``SMSEMOAReplacement`` (``calculateReferencePoint`` and ``createJointReferencePoint``)
takes, for each objective, the worst value of the population and the new child, and multiplies it
by 1.1, or divides it by 1.1 when it is negative, so that the reference lies above that value. It
recomputes the point at every step and has no parameter.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vamos
from vamos.engine.algorithm.config import SMSEMOAConfig
from vamos.engine.algorithm.smsemoa.helpers import jmetal_reference_point, refresh_selection_cache, survival_selection
from vamos.engine.algorithm.smsemoa.state import SMSEMOAState
from vamos.foundation.kernel.numpy_backend import NumPyKernel

# A front of four points and a non-dominated child.
FRONT = np.array([[0.0, 1.0], [0.1, 0.6], [0.5, 0.4], [1.0, 0.0]])
CHILD = np.array([[0.3, 0.45]])


def _builder():
    return SMSEMOAConfig.builder().pop_size(4).crossover("sbx", prob=0.9, eta=20.0).mutation("pm", prob=0.5, eta=20.0).selection("random")


def test_the_jmetal_reference_multiplies_the_worst_values_by_1_1() -> None:
    F = np.array([[1.0, -4.0, 0.0], [3.0, -2.0, -1.0], [2.0, -1.0, -3.0]])

    np.testing.assert_allclose(jmetal_reference_point(F), [3.3, -1.0 / 1.1, 0.0])


def test_the_builder_records_the_jmetal_rule() -> None:
    config = _builder().reference_point(rule="jmetal").build()

    assert config.reference_point == {"vector": None, "adaptive": True, "rule": "jmetal"}
    assert SMSEMOAConfig.from_dict(config.to_dict()).reference_point == config.reference_point


def test_the_default_reference_point_is_unchanged() -> None:
    assert SMSEMOAConfig.default(n_var=30).reference_point == {"vector": None, "offset": 1.0, "adaptive": True}
    assert _builder().reference_point(offset=0.5).build().reference_point == {
        "vector": None,
        "offset": 0.5,
        "adaptive": True,
    }


@pytest.mark.parametrize(
    "arguments",
    [{"vector": [2.0, 2.0]}, {"offset": 0.1}, {"adaptive": False}],
    ids=["vector", "offset", "fixed"],
)
def test_the_jmetal_rule_takes_no_other_setting(arguments: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="jmetal"):
        _builder().reference_point(rule="jmetal", **arguments)


def test_an_unknown_rule_is_rejected() -> None:
    with pytest.raises(ValueError, match="'relative', 'jmetal'"):
        _builder().reference_point(rule="nadir")


def _state(rule: str) -> SMSEMOAState:
    state = SMSEMOAState(
        X=np.arange(FRONT.shape[0], dtype=float).reshape(-1, 1),
        F=FRONT.copy(),
        G=None,
        rng=np.random.default_rng(0),
        pop_size=FRONT.shape[0],
        offspring_size=1,
        constraint_mode="none",
        ref_point=FRONT.max(axis=0) + 1.0,
        ref_offset=1.0,
        ref_adaptive=True,
        ref_rule=rule,
    )
    refresh_selection_cache(state, NumPyKernel())
    return state


@pytest.mark.parametrize(("rule", "removed"), [("relative", 2), ("jmetal", 0)])
def test_survival_measures_the_contributions_from_the_rules_reference(rule: str, removed: int) -> None:
    state = _state(rule)

    survival_selection(state, X_child=np.array([[9.0]]), F_child=CHILD, G_child=None, kernel=NumPyKernel())

    joint = np.vstack([FRONT, CHILD])
    expected = jmetal_reference_point(joint) if rule == "jmetal" else joint.max(axis=0) + (joint.max(axis=0) - joint.min(axis=0))
    np.testing.assert_allclose(state.ref_point, expected)
    # Close to the front, jMetal's reference shrinks the extremes' contributions, so an extreme point leaves.
    assert float(removed) not in state.X[:, 0]
    assert 9.0 in state.X[:, 0]


def test_a_run_with_the_jmetal_rule_records_it_and_replays_exactly(tmp_path: Path) -> None:
    config = (
        SMSEMOAConfig.builder()
        .pop_size(8)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob=1.0 / 6.0, eta=20.0)
        .selection("random")
        .reference_point(rule="jmetal")
        .build()
    )
    result = vamos.optimize("zdt1", algorithm="smsemoa", algorithm_config=config, max_evaluations=40, n_var=6, seed=3)
    relative = vamos.optimize(
        "zdt1", algorithm="smsemoa", algorithm_config=SMSEMOAConfig.default(pop_size=8, n_var=6), max_evaluations=40, n_var=6, seed=3
    )

    assert result.explain_defaults()["resolved_spec"]["algorithm"]["config"]["reference_point"]["rule"] == "jmetal"
    assert not np.array_equal(np.sort(result.F, axis=0), np.sort(relative.F, axis=0))
    source = vamos.save_result(result, tmp_path / "source")
    assert vamos.reproduce(source.root, output=tmp_path / "replay").exact


@pytest.mark.parametrize("engine", ["numpy", "numba"])
def test_the_compiled_step_gives_the_same_run(engine: str) -> None:
    if engine == "numba":
        pytest.importorskip("numba")
    config = _builder().reference_point(rule="jmetal").build()
    result = vamos.optimize("zdt1", algorithm="smsemoa", algorithm_config=config, max_evaluations=60, n_var=6, seed=5, engine=engine)
    reference = vamos.optimize("zdt1", algorithm="smsemoa", algorithm_config=config, max_evaluations=60, n_var=6, seed=5, engine="numpy")

    np.testing.assert_array_equal(result.F, reference.F)
