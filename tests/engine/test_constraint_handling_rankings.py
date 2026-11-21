"""ToR (two_rankings) and ShiP (shift_penalty), checked against the worked examples of their papers.

- ToR: Z. Ma, Y. Wang and W. Song, IEEE TSMC: Systems 51(8):5005-5016, 2021, Tables I-III, ten solutions A-J
  at P_fea = 0, 0.4 and 0.8. The tables give each solution's objectives, its level by the constrained-domination
  principle (CDP) and its positions R^c and R^p, but not its violation: the violations here are chosen to give
  those levels, and only their order matters.
- ShiP: Z. Ma and Y. Wang, IEEE TCYB 53(1):18-30, 2023, Table I, ten solutions A-J at P_fea = 0.3 and c = 2.

``test_constraint_handling_techniques`` and ``test_constraint_handling_algorithms`` run both techniques in every
algorithm that takes them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from vamos.engine.algorithm.components.constrained_ranking import pareto_ranks
from vamos.engine.algorithm.components.constraint_handling import make_constraint_handling, normalize_constraint_handling
from vamos.engine.algorithm.components.constraint_penalties import ShiftPenalty, shift_penalty_objectives
from vamos.engine.algorithm.components.constraint_two_rankings import TwoRankings, two_rankings_fitness, two_rankings_ranks
from vamos.engine.algorithm.nsgaii.helpers import compute_crowding, fronts_from_ranks, select_nsga2
from vamos.engine.tuning.grammar import builtin_parameter_space
from vamos.engine.tuning.racing.bridge_assignments import config_from_assignment
from vamos.foundation.kernel.registry import resolve_kernel

ENGINES = ["numpy", "numba"]
NAMES = "ABCDEFGHIJ"


def _kernel(engine: str) -> Any:
    if engine == "numba":
        pytest.importorskip("numba")
    return resolve_kernel(engine)


def _paper_crowding(F: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """The crowding distance of the paper's tables: the neighbours' differences in each objective, summed without
    dividing them by the objective's range."""
    n = F.shape[0]
    if n <= 2:
        return np.full(n, np.inf)
    distance = np.zeros(n)
    for m in range(F.shape[1]):
        order = np.argsort(F[:, m], kind="mergesort")
        values = F[order, m]
        if values[-1] - values[0] <= 0.0:
            continue
        distance[order[0]] = distance[order[-1]] = np.inf
        distance[order[1:-1]] += values[2:] - values[:-2]
    return distance


def _constraints(violations: list[float]) -> np.ndarray[Any, Any]:
    """One constraint per row: -1 (feasible) for a violation of 0, else the violation."""
    values = np.asarray(violations, dtype=float)
    return np.where(values > 0.0, values, -1.0)[:, None]


# ---- ToR: the paper's Tables I-III ----------------------------------------------------

TABLES: dict[str, dict[str, Any]] = {
    # P_fea = 0, alpha = 0.5: every solution is infeasible, so CDP puts each in a level of its own, in the order
    # of its violation, and R^c is that order.
    "I": {
        "F": [(0.05, 1.128), (0.249, 1.25), (0.495, 1.278), (0.668, 1.072), (0.823, 1.177),
              (0.9282, 0.925), (1.078, 0.981), (1.229, 0.583), (1.521, 0.386), (1.368, 0.055)],
        "violations": [7, 9, 6, 2, 5, 1, 4, 3, 10, 8],
        "R^c": [7, 9, 6, 2, 5, 1, 4, 3, 10, 8],
        "R^p": [1, 6, 10, 4, 9, 5, 8, 3, 7, 2],
        "alpha": 0.5,
        "best": "ADFHJ",
    },
    # P_fea = 0.4, alpha = 0.7: A, E, F and G are feasible; J, H, D, I, C and B follow in that order of violation.
    "II": {
        "F": [(0.034, 0.845), (0.195, 0.819), (0.278, 0.971), (0.296, 0.754), (0.415, 0.862),
              (0.574, 0.637), (0.764, 0.665), (0.865, 0.263), (1.018, 0.195), (1.1, 0.036)],
        "violations": [0, 6, 5, 3, 0, 0, 0, 2, 4, 1],
        "R^c": [1, 10, 9, 7, 3, 2, 4, 6, 8, 5],
        "R^p": [1, 7, 8, 5, 10, 3, 9, 4, 6, 2],
        "fit": [1, 9.1, 8.7, 6.4, 5.1, 2.3, 5.5, 5.4, 7.4, 4.1],
        "alpha": 0.7,
        "best": "AEFHJ",
    },
    # P_fea = 0.8, alpha = 0.9: all but B and I are feasible, and I is less violated than B.
    "III": {
        "F": [(0.035, 0.829), (0.164, 0.791), (0.415, 0.748), (0.523, 0.63), (0.575, 0.597),
              (0.675, 0.517), (0.8, 0.38), (0.882, 0.334), (0.961, 0.137), (1.014, 0.05)],
        "violations": [0, 2, 0, 0, 0, 0, 0, 0, 1, 0],
        "R^c": [1, 10, 3, 7, 8, 5, 6, 4, 9, 2],
        "R^p": [1, 4, 3, 9, 10, 5, 8, 7, 6, 2],
        "fit": [1, 9.4, 3, 7.2, 8.2, 5, 6.2, 4.3, 8.7, 2],
        "alpha": 0.9,
        "best": "ACFHJ",
    },
}  # fmt: skip


def _table(name: str) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], dict[str, Any]]:
    table = TABLES[name]
    return np.array(table["F"]), _constraints(table["violations"]), table


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", list(TABLES))
def test_tor_reproduces_the_positions_and_fitness_of_the_tables(name: str, engine: str) -> None:
    F, G, table = _table(name)

    constrained, pareto, fitness = two_rankings_fitness(_kernel(engine), F, G, _paper_crowding)

    assert constrained.tolist() == table["R^c"]
    assert pareto.tolist() == table["R^p"]
    alpha = table["alpha"]
    assert fitness == pytest.approx(alpha * np.array(table["R^c"]) + (1 - alpha) * np.array(table["R^p"]), abs=1e-12)
    if "fit" in table:
        assert fitness == pytest.approx(table["fit"], abs=1e-12)


def test_table_one_prints_its_fitness_with_the_weight_misprinted() -> None:
    # Table I prints 0.5 R^c + 0.4 R^p rather than eq. (6)'s 0.5 R^c + 0.5 R^p; the five best are the same.
    printed = [3.9, 6.9, 7, 2.6, 6.1, 2.5, 5.2, 2.7, 7.8, 4.8]
    table = TABLES["I"]
    assert printed == pytest.approx(0.5 * np.array(table["R^c"]) + 0.4 * np.array(table["R^p"]), abs=1e-12)
    assert "".join(sorted(NAMES[i] for i in np.argsort(printed, kind="stable")[:5])) == table["best"]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", list(TABLES))
def test_tor_selects_the_five_best_of_the_tables(name: str, engine: str) -> None:
    F, G, table = _table(name)
    kernel = _kernel(engine)

    for crowding in (_paper_crowding, None):
        ranks = two_rankings_ranks(kernel, F, G, crowding)
        assert sorted(ranks.tolist()) == list(range(10))
        assert "".join(sorted(NAMES[i] for i in np.flatnonzero(ranks < 5))) == table["best"]


def test_equal_fitness_is_ordered_by_the_constrained_position() -> None:
    # Table I with eq. (6): D, F and H all have fitness 3, and their R^c are 2, 1 and 3.
    F, G, _ = _table("I")

    ranks = two_rankings_ranks(resolve_kernel("numpy"), F, G, _paper_crowding)

    assert [NAMES[i] for i in np.argsort(ranks)] == list("FDHAJGEBCI")


@pytest.mark.parametrize("engine", ENGINES)
def test_vamos_crowding_normalizes_the_differences(engine: str) -> None:
    # Dividing each difference by the objective's range swaps D and F in Table I's first Pareto level, and orders
    # F, I and B differently in Table III's; R^c and Table II do not change.
    kernel = _kernel(engine)
    expected = {"I": [1, 6, 10, 5, 9, 4, 8, 3, 7, 2], "II": TABLES["II"]["R^p"], "III": [1, 6, 3, 9, 10, 4, 8, 7, 5, 2]}
    for name, pareto_positions in expected.items():
        F, G, table = _table(name)
        constrained, pareto, _ = two_rankings_fitness(kernel, F, G)
        assert constrained.tolist() == table["R^c"]
        assert pareto.tolist() == pareto_positions


@pytest.mark.parametrize("engine", ENGINES)
def test_the_technique_ranks_one_solution_per_front_and_nsgaii_keeps_the_best(engine: str) -> None:
    kernel = _kernel(engine)
    F, G, table = _table("II")

    ranks = TwoRankings().ranks(kernel, F, G, np.random.default_rng(0))
    fronts = fronts_from_ranks(ranks)
    kept = select_nsga2(fronts, compute_crowding(F, fronts), 5)

    assert all(len(front) == 1 for front in fronts)
    assert "".join(sorted(NAMES[i] for i in kept)) == table["best"]


def test_tor_has_no_parameters() -> None:
    assert isinstance(make_constraint_handling("two_rankings"), TwoRankings)
    with pytest.raises(ValueError, match="unknown: \\['constant'\\]"):
        normalize_constraint_handling(("two_rankings", {"constant": 1.0}))


def test_a_feasible_population_ranks_by_pareto_dominance_then_crowding() -> None:
    # With P_fea = 1, alpha = 1 and the fitness is R^c, which is then R^p.
    kernel = resolve_kernel("numpy")
    rng = np.random.default_rng(3)
    F, G = rng.random((25, 2)), -rng.random((25, 1))

    constrained, pareto, fitness = two_rankings_fitness(kernel, F, G)

    assert constrained.tolist() == pareto.tolist() == fitness.astype(int).tolist()
    assert np.all(np.diff(pareto_ranks(kernel, F)[np.argsort(constrained)]) >= 0)


# ---- ShiP: the paper's Table I ----------------------------------------------------------

SHIP_F = np.array([
    (0.172, 1.224), (0.299, 1.125), (0.362, 0.878), (0.528, 0.875), (0.632, 0.753),
    (0.733, 0.651), (0.819, 0.558), (0.937, 0.407), (1.014, 0.041), (1.351, 0.128),
])  # fmt: skip
# A, F and G are feasible, so P_fea = 0.3; the others have the table's violations.
SHIP_CV = np.array([0.0, 0.108, 0.045, 0.187, 0.134, 0.0, 0.0, 0.129, 0.101, 0.245])
SHIP_INFEASIBLE = [1, 2, 3, 4, 7, 8, 9]
SHIP_NADIR = [(0.733, 1.224)] * 4 + [(1.351, 0.558)] * 3
SHIP_SHIFT = [(0.261, 0.059), (0.223, 0.207), (0.123, 0.209), (0.061, 0.283), (0.248, 0.090), (0.202, 0.310), (0.000, 0.258)]
SHIP_MODIFIED = [(0.668, 1.293), (0.630, 1.130), (0.838, 1.271), (0.827, 1.170), (1.314, 0.626), (1.317, 0.452), (1.596, 0.631)]


def test_ship_reproduces_the_table() -> None:
    G = _constraints(SHIP_CV.tolist())

    modified = shift_penalty_objectives(SHIP_F, G, 2.0)

    rows = SHIP_INFEASIBLE
    shift = modified[rows] - SHIP_F[rows] - SHIP_CV[rows, None]
    # The shift is c P_fea (z* - f) = 0.6 (z* - f), so it gives the local feasible nadir back exactly.
    assert SHIP_F[rows] + shift / 0.6 == pytest.approx(np.array(SHIP_NADIR), abs=1e-12)
    # The table rounds the objectives to three decimals and computes the rest before rounding.
    assert shift == pytest.approx(np.array(SHIP_SHIFT), abs=1e-3)
    assert modified[rows] == pytest.approx(np.array(SHIP_MODIFIED), abs=1e-3)
    feasible = [0, 5, 6]
    assert np.array_equal(modified[feasible], SHIP_F[feasible])


@pytest.mark.parametrize("engine", ENGINES)
def test_only_c_and_i_join_the_feasible_solutions_in_the_first_front(engine: str) -> None:
    # "only C and I are nondominated with the three feasible solutions" (section III-C, Fig. 3(b)).
    ranks = ShiftPenalty(2.0).ranks(_kernel(engine), SHIP_F, _constraints(SHIP_CV.tolist()), np.random.default_rng(0))

    assert "".join(NAMES[i] for i in np.flatnonzero(ranks == 0)) == "ACFGI"


def test_the_local_feasible_nadir_takes_strictly_greater_values() -> None:
    # u = (0.5, 0.5); feasible (0.5, 0.9) is not above it in f1, and (0.7, 0.6) is: z* = (0.7, 0.6). The virtual
    # extreme value is the largest of all the rows, the infeasible ones included.
    F = np.array([[0.5, 0.9], [0.7, 0.6], [0.5, 0.5], [0.9, 0.95]])
    G = _constraints([0, 0, 0.1, 0.2])

    modified = shift_penalty_objectives(F, G, 1.0)  # c P_fea = 0.5

    assert modified[2] == pytest.approx([0.5 + 0.5 * 0.2 + 0.1, 0.5 + 0.5 * 0.1 + 0.1])
    # (0.9, 0.95) has no feasible value above it: z* is the largest of the rows, its own, so only the penalty moves it.
    assert modified[3] == pytest.approx([0.9 + 0.2, 0.95 + 0.2])


def test_ship_keeps_the_objectives_without_feasible_solutions() -> None:
    F = np.array([[0.1, 0.9], [0.9, 0.1], [0.5, 0.5]])
    infeasible = _constraints([0.3, 0.1, 0.2])
    feasible = _constraints([0, 0, 0])

    assert np.array_equal(shift_penalty_objectives(F, infeasible, 2.0), F)
    assert np.array_equal(shift_penalty_objectives(F, feasible, 2.0), F)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (("shift_penalty", {"constant": 0.0}), "constant must lie in"),
        (("shift_penalty", {}), "missing: \\['constant'\\]"),
    ],
)
def test_invalid_ship_parameters_are_rejected(value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_constraint_handling(value)


# ---- Tuning spaces ----------------------------------------------------------------------


@pytest.mark.parametrize("space_name", ["nsgaii_constrained", "nsgaiii_constrained", "smsemoa_constrained", "agemoea_constrained"])
def test_the_constrained_spaces_offer_ship_and_nsgaii_tor(space_name: str) -> None:
    space = builtin_parameter_space(space_name)
    technique = space.parameter("constraint_handling")
    assert ("two_rankings" in technique.values) == (space_name == "nsgaii_constrained")
    assert [child.name for child in technique.production("shift_penalty")] == ["shift_penalty_constant"]
    constant = space.parameter("shift_penalty_constant")
    assert (constant.range, constant.default) == ((1.0, 5.0), 2.0)

    defaults = dict(space.default_values())
    ship = {**defaults, "constraint_handling": "shift_penalty", "shift_penalty_constant": 3.0}
    tor = {**defaults, "constraint_handling": "two_rankings"}

    assert config_from_assignment(space_name, ship).constraint_handling == ("shift_penalty", {"constant": 3.0})
    if space_name == "nsgaii_constrained":
        assert technique.production("two_rankings") == ()
        assert config_from_assignment(space_name, tor).constraint_handling == ("two_rankings", {})
    else:
        with pytest.raises(ValueError, match="defined for NSGA-II only"):
            config_from_assignment(space_name, tor)
