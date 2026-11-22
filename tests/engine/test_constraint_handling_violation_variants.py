"""Dam and Truong's options of violation_objective (IJCAS 22(10):3240-3252, 2024), on hand-computed populations.

The paper's Table 1 reports runs, not rankings, so these tests check Defs. 6 and 7, the CDF of section 3.2 and
Algorithm A on small populations computed by hand.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np
import pytest

from vamos import optimize
from vamos.algorithms import NSGAIIConfig, NSGAIIIConfig
from vamos.engine.algorithm.components import constraint_handling
from vamos.engine.algorithm.components.constrained_ranking import pareto_ranks, violation
from vamos.engine.algorithm.components.constraint_handling import make_constraint_handling, normalize_constraint_handling
from vamos.engine.algorithm.components.constraint_penalties import ViolationObjective, violation_objectives
from vamos.engine.tuning.grammar import builtin_parameter_space
from vamos.engine.tuning.racing.bridge_assignments import config_from_assignment
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.problems import LIRCMOP

ENGINES = ["numpy", "numba"]


def _kernel(engine: str) -> Any:
    if engine == "numba":
        pytest.importorskip("numba")
    return resolve_kernel(engine)


def _config(params: dict[str, Any], **schedule: Any) -> NSGAIIConfig:
    config = (
        NSGAIIConfig.builder()
        .pop_size(30)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .build()
    )
    return dataclasses.replace(config, constraint_handling=("violation_objective", params), **schedule)


def _run(params: dict[str, Any], **schedule: Any) -> Any:
    return optimize(LIRCMOP(2, n_var=10), algorithm="nsgaii", algorithm_config=_config(params, **schedule), max_evaluations=1500, seed=6)


# ---- The defaults rank as before ---------------------------------------------------------------


def test_the_defaults_rank_by_the_objectives_and_the_overall_violation() -> None:
    kernel = resolve_kernel("numpy")
    rng = np.random.default_rng(8)
    for _ in range(10):
        F, G = rng.random((30, 2)), rng.random((30, 3)) - 0.6
        expected = pareto_ranks(kernel, np.column_stack([F, violation(G)]))
        for technique in (ViolationObjective(), make_constraint_handling(("violation_objective", {"violation_norm": "l1"}))):
            assert technique.ranks(kernel, F, G, rng).tolist() == expected.tolist()


def test_explicit_defaults_give_the_run_without_options() -> None:
    plain = _run({})
    explicit = _run({"violation_norm": "l1", "ratio_objective": False, "feasible_first": False})

    assert np.array_equal(plain.X, explicit.X) and np.array_equal(plain.F, explicit.F)


# ---- Def. 7: the norm of the violation ------------------------------------------------------------

G3 = np.array([[0.5, 2.0, -1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 0.0]])


@pytest.mark.parametrize(("norm", "expected"), [("l1", [2.5, 2.0, 0.0]), ("l2_squared", [4.25, 2.0, 0.0]), ("linf", [2.0, 1.0, 0.0])])
def test_the_violation_is_a_norm_to_its_order(norm: str, expected: list[float]) -> None:
    assert violation_objectives(G3, norm)[:, 0].tolist() == expected


# ---- Def. 6 and section 3.2: the constraint violation ratio and its CDF weights ---------------------

# Constraint 1 is violated by two rows, constraints 2 and 3 by one each: f = (2, 1, 1).
G4 = np.array([[1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0], [-1.0, -1.0, -1.0]])


def test_equal_weights_give_the_share_of_violated_constraints() -> None:
    assert violation_objectives(G4, "l1", True)[:, 0] == pytest.approx([1 / 3, 2 / 3, 1 / 3, 0])


def test_the_cdf_weighs_the_constraints_that_the_population_violates_most() -> None:
    # alpha = 0.5: w* = (1 + 0.5 * 2/4, 1 + 0.5 * 1/4, 1 + 0.5 * 1/4) = (1.25, 1.125, 1.125), summing to 3.5.
    ratio = violation_objectives(G4, "l1", True, 0.5)[:, 0]

    assert ratio == pytest.approx([1.25 / 3.5, 2.375 / 3.5, 1.125 / 3.5, 0])
    # Property 3: a constraint that only it violates weighs 1 + alpha.
    single = np.array([[1.0, -1.0, -1.0], [2.0, -1.0, -1.0], [-1.0, -1.0, -1.0]])
    assert violation_objectives(single, "l1", True, 0.5)[:, 0] == pytest.approx([1.5 / 3.5, 1.5 / 3.5, 0])


def test_the_ratio_is_one_more_objective_before_the_violation() -> None:
    kernel = resolve_kernel("numpy")
    F = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    # The first row violates one constraint by 3, the second both by 0.5. Over [F, V] the second dominates the first
    # (V 1 against 3); over [F, R, V] it does not, since it violates more constraints (R 1 against 0.5).
    G = np.array([[3.0, -1.0], [0.5, 0.5], [-1.0, -1.0]])

    assert violation_objectives(G, "l1", True)[:, 0].tolist() == [0.5, 1.0, 0.0]
    assert ViolationObjective().ranks(kernel, F, G, np.random.default_rng(0)).tolist() == [1, 0, 0]
    assert ViolationObjective(ratio_objective=True).ranks(kernel, F, G, np.random.default_rng(0)).tolist() == [0, 0, 0]
    assert ViolationObjective("linf", ratio_objective=True).ranks(kernel, F, G, np.random.default_rng(0)).tolist() == [0, 0, 0]


# ---- Algorithm A: the feasible-first survival ----------------------------------------------------

# Feasible a, b, c, d: a, b and c are non-dominated, a and b at its ends, and d is dominated. Infeasible, with two
# constraints: e (R 0.5, V 0.2), f (R 1, V 0.2), g (R 0.5, V 0.5) and h (R 0.5, V 0.1).
FF = np.array([(0, 1), (1, 0), (0.5, 0.5), (0.9, 0.9), (0, 0), (0, 0), (0, 0), (0, 0)], dtype=float)
GF = np.array([(-1, -1), (-1, -1), (-1, -1), (-1, -1), (0.2, -1), (0.1, 0.1), (0.5, -1), (0.1, -1)], dtype=float)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(("ratio", "order"), [(True, "abcdhegf"), (False, "abcdhefg")])
def test_feasible_first_orders_the_infeasible_by_ratio_then_violation(engine: str, ratio: bool, order: str) -> None:
    # With the ratio objective, R decides first (h, e and g before f); without it, V alone (e and f tie at 0.2 and
    # keep the population's order).
    technique = ViolationObjective(ratio_objective=ratio, feasible_first=True)

    ranks = technique.ranks(_kernel(engine), FF, GF, np.random.default_rng(0))

    assert technique.total_order is True
    assert "".join("abcdefgh"[i] for i in np.argsort(ranks)) == order


def test_feasible_first_takes_feasibility_from_the_constraints_not_from_the_norm() -> None:
    # A violation of 1e-200 squares to 0, so its l2_squared V is 0; the row is infeasible all the same, and ranks last.
    F = np.array([[0.0, 0.0], [1.0, 1.0]])
    G = np.array([[1e-200], [-1.0]])
    technique = ViolationObjective("l2_squared", feasible_first=True)

    assert violation_objectives(G, "l2_squared")[:, 0].tolist() == [0.0, 0.0]
    assert technique.ranks(resolve_kernel("numpy"), F, G, np.random.default_rng(0)).tolist() == [1, 0]


def test_feasible_first_skips_crowding_bit_identically(monkeypatch: pytest.MonkeyPatch) -> None:
    params = {"ratio_objective": True, "cdf_alpha": 0.5, "feasible_first": True}
    skipped = _run(params)
    build = constraint_handling.make_constraint_handling

    def without_total_order(value: object) -> Any:
        technique = build(value)
        technique.total_order = False
        return technique

    monkeypatch.setattr(constraint_handling, "make_constraint_handling", without_total_order)
    computed = _run(params)

    assert np.array_equal(skipped.X, computed.X) and np.array_equal(skipped.F, computed.F)


def test_feasible_first_is_nsgaii_only() -> None:
    base = NSGAIIIConfig.default(pop_size=30, n_obj=2)
    with pytest.raises(ValueError, match="'violation_objective' with feasible_first is defined for NSGA-II only"):
        dataclasses.replace(base, constraint_handling=("violation_objective", {"feasible_first": True}))
    options = {"violation_norm": "linf", "ratio_objective": True, "cdf_alpha": 0.5}
    assert dataclasses.replace(base, constraint_handling=("violation_objective", options)).constraint_handling == (
        "violation_objective",
        options,
    )


# ---- Validation, configurations and tuning spaces ------------------------------------------------


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"violation_norm": "l3"}, "violation_norm must be one of l1, l2_squared, linf"),
        ({"ratio_objective": 1}, "ratio_objective must be True or False"),
        ({"feasible_first": "yes"}, "feasible_first must be True or False"),
        ({"cdf_alpha": 0.5}, "cdf_alpha weighs the ratio objective"),
        ({"ratio_objective": True, "cdf_alpha": -1.0}, "cdf_alpha must lie in"),
        ({"weights": [1, 2]}, "unknown: \\['weights'\\]"),
    ],
)
def test_invalid_options_are_rejected(params: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_constraint_handling(("violation_objective", params))


def test_only_the_given_options_are_kept_and_they_round_trip_through_json() -> None:
    assert normalize_constraint_handling(("violation_objective", {})) == ("violation_objective", {})
    config = _config({"violation_norm": "l2_squared", "ratio_objective": True, "cdf_alpha": 0.5, "feasible_first": True})

    restored = NSGAIIConfig.from_dict(json.loads(json.dumps(config.to_dict())))

    assert restored == config


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [("none", {}), ("equal", {"ratio_objective": True}), ("cdf", {"ratio_objective": True, "cdf_alpha": 0.7})],
)
def test_the_spaces_offer_the_options(ratio: str, expected: dict[str, Any]) -> None:
    for name in ("nsgaii_constrained", "nsgaiii_constrained", "smsemoa_constrained", "agemoea_constrained"):
        space = builtin_parameter_space(name)
        assignment = {
            **dict(space.default_values()),
            "constraint_handling": "violation_objective",
            "violation_objective_norm": "l2_squared",
            "violation_objective_ratio": ratio,
            "violation_objective_cdf_alpha": 0.7,
        }
        nsgaii = name == "nsgaii_constrained"
        if nsgaii:
            assignment["violation_objective_feasible_first"] = True
        params = {"violation_norm": "l2_squared", **expected, **({"feasible_first": True} if nsgaii else {})}
        assert config_from_assignment(name, assignment).constraint_handling == ("violation_objective", params)
        assert ("violation_objective_feasible_first" in space.names()) == nsgaii
        alpha = space.parameter("violation_objective_cdf_alpha")
        assert (alpha.range, alpha.default) == ((0.1, 2.0), 0.5)
