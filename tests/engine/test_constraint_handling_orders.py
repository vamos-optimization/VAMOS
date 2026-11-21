"""IDEA (infeasibility_driven) and ATM (adaptive_tradeoff), checked against the worked examples of their papers.

- IDEA: Ray, Singh, Isaacs and Smith, SCI 198, 2009, Table 1: the violation measure of ten solutions with three
  constraints. Its selection follows section 2's rule, checked on small populations.
- ATM: Wang, Cai, Zhou and Zeng, IEEE TEVC 12(1), 2008, the example of p. 84: ten single-objective solutions,
  five feasible, ordered for phi = 0, 0.5 and 1. Phase one follows Fig. 3; C is Li et al.'s eq. 3.

``test_constraint_handling_techniques`` runs both in NSGA-II, and ``test_constraint_handling_algorithms`` checks
that the other algorithms reject them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vamos.engine.algorithm.components.constrained_ranking import pareto_ranks
from vamos.engine.algorithm.components.constraint_handling import make_constraint_handling, normalize_constraint_handling
from vamos.engine.algorithm.components.constraint_orders import (
    AdaptiveTradeoff,
    InfeasibilityDriven,
    adaptive_tradeoff_fitness,
    adaptive_tradeoff_ranks,
    infeasibility_driven_ranks,
    normalized_violation,
    violation_measure,
)
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.engine.algorithm.nsgaii.run import _checkpoint_extra
from vamos.engine.tuning.grammar import builtin_parameter_space
from vamos.engine.tuning.racing.bridge_assignments import config_from_assignment
from vamos.foundation.eval.backends import SerialEvalBackend
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.problems import LIRCMOP

ENGINES = ["numpy", "numba"]


def _kernel(engine: str) -> Any:
    if engine == "numba":
        pytest.importorskip("numba")
    return resolve_kernel(engine)


def _order(ranks: np.ndarray[Any, Any]) -> list[int]:
    return np.argsort(ranks).tolist()


# ---- IDEA: Table 1 ------------------------------------------------------------------------

# The violations of constraints C1, C2 and C3 of solutions 1-10; 0 is a satisfied constraint.
IDEA_VIOLATIONS = np.array([
    (3.50, 90.60, 8.09), (5.76, 7.80, 6.70), (0.00, 3.40, 7.10), (1.25, 0.00, 0.69), (13.75, 90.10, 5.87),
    (100.70, 2.34, 3.20), (0.00, 5.09, 4.76), (1.90, 0.00, 0.00), (0.00, 0.56, 0.00), (8.90, 2.30, 9.80),
])  # fmt: skip
IDEA_RELATIVE_RANKS = [
    [3, 4, 0, 1, 6, 7, 0, 2, 0, 5],
    [8, 6, 4, 0, 7, 3, 5, 0, 1, 2],
    [7, 5, 6, 1, 4, 2, 3, 0, 0, 8],
]
IDEA_MEASURE = [18, 15, 10, 2, 17, 12, 8, 2, 1, 15]


def test_idea_reproduces_the_violation_measure_of_table_one() -> None:
    for column, ranks in enumerate(IDEA_RELATIVE_RANKS):
        assert violation_measure(IDEA_VIOLATIONS[:, [column]]).tolist() == ranks
    assert violation_measure(IDEA_VIOLATIONS).tolist() == IDEA_MEASURE


def test_equal_violations_share_a_rank_and_the_next_follows() -> None:
    # The paper does not say how ranks continue after a tie; VAMOS uses dense ranks.
    assert violation_measure(np.array([[0.5], [0.5], [1.0], [0.0], [-1.0]])).tolist() == [1, 1, 2, 0, 0]


# Four feasible solutions a-d and four infeasible ones e-h, with one constraint. Among the feasible ones, a, b and c
# are non-dominated (a and b at the ends) and d is dominated. Among the infeasible ones, with the violation measure
# as a third objective (e 2, f 3, g 1, h 4), e, g and h are non-dominated, all at an end of an objective, and e
# dominates f. So the feasible order is a, b, c, d and the infeasible one e, g, h, f.
IDEA_F = np.array([(0, 1), (1, 0), (0.5, 0.5), (0.8, 0.8), (0.1, 0.1), (0.2, 0.2), (2, 2), (0.05, 0.9)])
IDEA_G = np.array([[-1.0], [-1.0], [-1.0], [-1.0], [0.2], [0.5], [0.1], [1.0]])
NAMES = "abcdefgh"


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    ("proportion", "population", "rows", "first"),
    [
        # N = 4, N_inf = 1: e, then the feasible ones, then the other infeasible ones.
        (0.25, 4, range(8), "eabcdghf"),
        # N_inf = 0.3 * 5 = 1.5, rounded half up to 2.
        (0.3, 5, range(8), "egabcdhf"),
        # Only two feasible solutions: the selection of N = 4 fills with the next infeasible one, g.
        (0.25, 4, [0, 1, 4, 5, 6, 7], "eabghf"),
        # Fewer infeasible solutions than N_inf = 2: all of them, then the feasible ones.
        (0.5, 4, [0, 1, 2, 3, 4], "eabcd"),
    ],
)
def test_idea_ranks_the_best_infeasible_solutions_above_the_feasible_ones(
    engine: str, proportion: float, population: int, rows: Any, first: str
) -> None:
    rows = list(rows)

    ranks = infeasibility_driven_ranks(_kernel(engine), IDEA_F[rows], IDEA_G[rows], proportion, population)

    assert "".join(NAMES[rows[i]] for i in _order(ranks)) == first


def test_idea_keeps_the_proportion_of_its_example() -> None:
    # Section 2: of 200 solutions, 40 infeasible, IDEA with alpha = 0.2 keeps 20 infeasible and 80 feasible.
    rng = np.random.default_rng(1)
    F = rng.random((200, 2))
    G = np.where(np.arange(200)[:, None] < 40, rng.random((200, 1)) + 0.01, -1.0)

    ranks = infeasibility_driven_ranks(resolve_kernel("numpy"), F, G, 0.2, 100)

    kept = np.flatnonzero(ranks < 100)
    assert np.count_nonzero(kept < 40) == 20
    # The kept infeasible solutions rank above the feasible ones.
    assert sorted(ranks[:40][ranks[:40] < 100].tolist()) == list(range(20))


def test_idea_needs_the_population_size() -> None:
    technique = make_constraint_handling(("infeasibility_driven", {"proportion": 0.2}))
    assert isinstance(technique, InfeasibilityDriven) and technique.total_order

    with pytest.raises(ValueError, match="bind_run"):
        technique.ranks(resolve_kernel("numpy"), IDEA_F, IDEA_G, np.random.default_rng(0))
    technique.bind_run(4, 100)
    assert technique.ranks(resolve_kernel("numpy"), IDEA_F, IDEA_G, np.random.default_rng(0)).tolist() == [1, 2, 3, 4, 0, 7, 5, 6]


# ---- ATM: Wang et al.'s example (p. 84) -----------------------------------------------------

# (f, G) of x1-x10: x1-x5 feasible.
WANG_F = np.array([[1.0], [1.5], [2.0], [2.5], [3.0], [0.5], [1.2], [1.7], [2.2], [3.2]])
WANG_G = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.01], [0.03], [0.005], [0.001], [0.02]])
WANG_ORDERS = {
    0.0: [1, 2, 3, 4, 5, 9, 8, 6, 10, 7],
    0.5: [1, 2, 3, 9, 8, 4, 6, 5, 7, 10],
    1.0: [1, 2, 6, 3, 8, 9, 4, 5, 7, 10],
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("phi", list(WANG_ORDERS))
def test_atm_reproduces_the_orders_of_the_example(phi: float, engine: str) -> None:
    ranks = adaptive_tradeoff_ranks(_kernel(engine), WANG_F, WANG_G, phi)

    assert [i + 1 for i in _order(ranks)] == WANG_ORDERS[phi]


def test_atm_fitness_of_the_example() -> None:
    # Eqs. 7-10 by hand: f_min = 1 and f_max = 3 over x1-x5, and the violations range over [0.001, 0.03].
    fitness = adaptive_tradeoff_fitness(WANG_F, normalized_violation(WANG_G), 0.5)[:, 0]

    expected = [0, 0.5 / 2.2, 1 / 2.2, 1.5 / 2.2, 2 / 2.2]
    expected += [1 / 2.2 + 0.009 / 0.029, 1 / 2.2 + 1, 1 / 2.2 + 0.004 / 0.029, 1.2 / 2.2, 1 + 0.019 / 0.029]
    assert fitness == pytest.approx(expected, abs=1e-12)


def test_c_is_the_mean_of_the_violations_over_their_largest() -> None:
    # Li et al., eq. 3: C_j^max = (2, 4), and the third constraint, which no row violates, adds 0.
    G = np.array([[2.0, 0.0, -1.0], [1.0, 4.0, -1.0], [0.0, 2.0, -1.0]])

    assert normalized_violation(G).tolist() == pytest.approx([1 / 3, (0.5 + 1) / 3, 0.5 / 3])


@pytest.mark.parametrize("engine", ENGINES)
def test_atm_phase_one_selects_half_the_non_dominated_by_violation(engine: str) -> None:
    # (f, C) with no feasible row. The non-dominated r0, r1, r2 and r4 sort by C as r4, r2, r1, r0, and the first
    # half, r4 and r2, come first; then r1 of r0 and r1; then r3 of r0 and r3; then r0, and last r5.
    F = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [1.5]])
    G = np.array([[0.9], [0.5], [0.1], [0.6], [0.05], [1.0]])

    assert _order(adaptive_tradeoff_ranks(_kernel(engine), F, G, 0.0)) == [4, 2, 1, 3, 0, 5]


def test_atm_phase_one_puts_the_least_violated_first() -> None:
    # Wang et al.'s Property 1, on random infeasible populations with two objectives and three constraints.
    rng = np.random.default_rng(2)
    kernel = resolve_kernel("numpy")
    for _ in range(20):
        F, G = rng.random((30, 2)), rng.random((30, 3)) + 1e-3
        ranks = adaptive_tradeoff_ranks(kernel, F, G, 0.0)
        assert int(np.argmin(ranks)) == int(np.argmin(normalized_violation(G)))
        assert sorted(ranks.tolist()) == list(range(30))


def test_atm_phase_three_is_nsgaii_order() -> None:
    kernel = resolve_kernel("numpy")
    rng = np.random.default_rng(4)
    F, G = rng.random((25, 2)), -rng.random((25, 2))
    fronts, crowding = kernel.nsga2_ranking(F)

    ranks = adaptive_tradeoff_ranks(kernel, F, G, 0.3)

    assert _order(ranks) == np.lexsort((np.arange(25), -crowding, fronts)).tolist()
    assert np.all(np.diff(pareto_ranks(kernel, F)[_order(ranks)]) >= 0)


def test_atm_learns_phi_from_each_population_that_survives() -> None:
    # phi is 0 before the first replacement (Wang et al., Fig. 4), then the feasible share of the survivors.
    problem = LIRCMOP(2, n_var=10)
    config = {
        "pop_size": 20,
        "crossover": ("sbx", {"prob": 0.9, "eta": 20.0}),
        "mutation": ("pm", {"prob": "1/n", "eta": 20.0}),
        "selection": ("tournament", {"size": 2}),
        "constraint_handling": ("adaptive_tradeoff", {}),
    }
    algorithm = NSGAII(config, kernel=resolve_kernel("numpy"))
    algorithm.initialize(problem, ("max_evaluations", 400), seed=5)
    state = algorithm._st
    assert state is not None
    technique = state.constraint_handling
    assert isinstance(technique, AdaptiveTradeoff) and technique.phi == 0.0
    backend = SerialEvalBackend()
    while not algorithm.should_terminate():
        algorithm.tell(backend.evaluate(algorithm.ask(), problem))
        assert state.G is not None
        assert technique.phi == np.count_nonzero(np.all(state.G <= 0.0, axis=1)) / state.G.shape[0]
        assert _checkpoint_extra(state)["constraint_handling"] == {"phi": technique.phi}

    restored = AdaptiveTradeoff()
    restored.restore(technique.state())
    assert restored.phi == technique.phi


# ---- Validation and tuning spaces -------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (("infeasibility_driven", {"proportion": 0.0}), "proportion must lie in"),
        (("infeasibility_driven", {"proportion": 1.5}), "proportion must lie in"),
        (("infeasibility_driven", {}), "missing: \\['proportion'\\]"),
        (("adaptive_tradeoff", {"proportion": 0.2}), "unknown: \\['proportion'\\]"),
    ],
)
def test_invalid_parameters_are_rejected(value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_constraint_handling(value)


def test_only_the_nsgaii_space_offers_idea_and_atm() -> None:
    space = builtin_parameter_space("nsgaii_constrained")
    technique = space.parameter("constraint_handling")
    assert [child.name for child in technique.production("infeasibility_driven")] == ["infeasibility_driven_proportion"]
    assert technique.production("adaptive_tradeoff") == ()
    proportion = space.parameter("infeasibility_driven_proportion")
    assert (proportion.range, proportion.default) == ((0.05, 0.5), 0.2)

    defaults = dict(space.default_values())
    idea = {**defaults, "constraint_handling": "infeasibility_driven", "infeasibility_driven_proportion": 0.3}
    atm = {**defaults, "constraint_handling": "adaptive_tradeoff"}
    assert config_from_assignment("nsgaii_constrained", idea).constraint_handling == ("infeasibility_driven", {"proportion": 0.3})
    assert config_from_assignment("nsgaii_constrained", atm).constraint_handling == ("adaptive_tradeoff", {})

    for name in ("nsgaiii_constrained", "smsemoa_constrained", "agemoea_constrained"):
        values = builtin_parameter_space(name).parameter("constraint_handling").values
        assert "infeasibility_driven" not in values and "adaptive_tradeoff" not in values
        with pytest.raises(ValueError, match="defined for NSGA-II only"):
            config_from_assignment(name, {**dict(builtin_parameter_space(name).default_values()), **atm})
