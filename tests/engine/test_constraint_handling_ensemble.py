"""MCR (multiple_constraint_ranking) and the ensemble, NSGA-II only.

- MCR: Garcia, de Lima, Lemonge and Jacob, Comput. Struct. 187:77-87, 2017, Tables 1-2: five single-objective
  individuals with three constraints, their ranks R_f, R_Nv and R_phi^j, and their fitness F (eq. 10).
- The ensemble: Vodopija, Oyama and Filipic, GECCO '19 Companion, pp. 2072-2075, in Fukumoto and Oyama's framework
  (EvoApplications 2018); checked through its members, its weights and NSGA-II runs.

``test_constraint_handling_techniques``, ``_algorithms`` and ``_survival`` run both in NSGA-II, check that the other
algorithms reject them and that skipping the crowding distance is bit-identical.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from vamos.engine.algorithm.components.constrained_ranking import pareto_ranks
from vamos.engine.algorithm.components.constraint_ensemble import (
    Ensemble,
    MultipleConstraintRanking,
    competition_ranks,
    multiple_constraint_ranking_fitness,
    nsgaii_fitness,
    objective_ranks,
)
from vamos.engine.algorithm.components.constraint_handling import make_constraint_handling, normalize_constraint_handling
from vamos.engine.tuning.grammar import builtin_parameter_space
from vamos.engine.tuning.racing.bridge_assignments import config_from_assignment
from vamos.foundation.kernel.registry import resolve_kernel

ENGINES = ["numpy", "numba"]


def _kernel(engine: str) -> Any:
    if engine == "numba":
        pytest.importorskip("numba")
    return resolve_kernel(engine)


# ---- MCR: Garcia et al.'s Tables 1-2 -------------------------------------------------------------

GARCIA_F = np.array([[5.41], [1.13], [8.7], [2.0], [10.0]])
# The violations v_1, v_2 and v_3 of I1-I5; I5 is the only feasible individual.
GARCIA_V = np.array([[6.24, 3.4e5, 0.002], [7.8, 0.0, 0.03], [3.1, 4.3e5, 0.0012], [0.0, 0.0, 0.04], [0.0, 0.0, 0.0]])
GARCIA_RANKS = {
    "R_f": [3, 1, 4, 2, 5],
    "R_Nv": [4, 3, 4, 2, 1],
    "R_phi": [[4, 5, 3, 1, 1], [4, 1, 5, 1, 1], [3, 4, 2, 5, 1]],
}
GARCIA_FITNESS = [18, 14, 18, 11, 9]


@pytest.mark.parametrize("engine", ENGINES)
def test_mcr_reproduces_garcia_et_al_tables(engine: str) -> None:
    kernel = _kernel(engine)

    assert objective_ranks(kernel, GARCIA_F).tolist() == GARCIA_RANKS["R_f"]
    assert competition_ranks(np.count_nonzero(GARCIA_V > 0.0, axis=1)).tolist() == GARCIA_RANKS["R_Nv"]
    assert [competition_ranks(column).tolist() for column in GARCIA_V.T] == GARCIA_RANKS["R_phi"]
    # A feasible individual exists, so F adds every rank (eq. 10, second case): I5 is the best, then I4.
    assert multiple_constraint_ranking_fitness(kernel, GARCIA_F, GARCIA_V).tolist() == GARCIA_FITNESS
    ranks = MultipleConstraintRanking().ranks(kernel, GARCIA_F, GARCIA_V, np.random.default_rng(0))
    assert [f"I{i + 1}" for i in np.argsort(ranks)] == ["I5", "I4", "I2", "I1", "I3"]


def test_without_feasible_individuals_mcr_ignores_the_objective() -> None:
    # Eq. 10, first case, on I1-I4: F = R_Nv + sum_j R_phi^j, ranked among the four.
    F, V = GARCIA_F[:4], GARCIA_V[:4]
    expected = competition_ranks(np.count_nonzero(V > 0.0, axis=1)) + sum(competition_ranks(column) for column in V.T)

    assert multiple_constraint_ranking_fitness(resolve_kernel("numpy"), F, V).tolist() == expected.tolist()


@pytest.mark.parametrize("engine", ENGINES)
def test_equal_objectives_share_a_rank(engine: str) -> None:
    # Three equal values share a level; their crowding distances must not part them (Garcia et al.: equal values,
    # equal rank). With two objectives, duplicated rows share the best fitness among them.
    kernel = _kernel(engine)

    assert objective_ranks(kernel, np.array([[1.0], [1.0], [1.0], [2.0]])).tolist() == [1, 1, 1, 4]
    duplicated = np.array([[0.0, 1.0], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [1.0, 0.0]])
    ranks = objective_ranks(kernel, duplicated)
    assert ranks[1] == ranks[2] == ranks[3]


# ---- The NSGA-II fitness of the framework ----------------------------------------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_the_nsgaii_fitness_keeps_the_pareto_order_and_prefers_the_extremes(engine: str) -> None:
    kernel = _kernel(engine)
    # One front: a and b at its ends (infinite crowding), c and e inside with distances 1.2 and 1.5; and d alone in
    # the second front.
    F = np.array([(0.0, 1.0), (1.0, 0.0), (0.25, 0.75), (0.9, 0.9), (0.6, 0.4)])

    fitness = nsgaii_fitness(kernel, F, pareto_ranks(kernel, F))

    # Finite distances span [0, 1/4] and infinite ones are 1/2: a, b -0.5; e -0.25; c 0; d 1 - 0.5.
    assert fitness.tolist() == pytest.approx([-0.5, -0.5, 0.0, 0.5, -0.25])


def test_equal_finite_distances_scale_to_zero() -> None:
    kernel = resolve_kernel("numpy")
    # c is the only finite distance of its front; it stays below the extremes a and b.
    F = np.array([(0.0, 1.0), (1.0, 0.0), (0.5, 0.5)])

    assert nsgaii_fitness(kernel, F, pareto_ranks(kernel, F)).tolist() == [-0.5, -0.5, 0.0]


# ---- The ensemble ----------------------------------------------------------------------------------

rng = np.random.default_rng(12)
ENSEMBLE_F = rng.random((24, 2))
ENSEMBLE_G = np.column_stack([rng.random(24) - 0.5, rng.random(24) - 0.7])


def _ensemble(weights: tuple[float, float, float, float], progress: float = 0.5) -> Ensemble:
    technique = Ensemble(weights)
    technique.bind_run(12, 1200)
    technique.update(resolve_kernel("numpy"), ENSEMBLE_F, ENSEMBLE_G, progress)
    return technique


def test_each_member_is_a_quality_in_zero_one() -> None:
    qualities = _ensemble((1, 1, 1, 1)).qualities(resolve_kernel("numpy"), ENSEMBLE_F, ENSEMBLE_G)

    assert qualities.shape == (24, 4)
    assert qualities.min(axis=0).tolist() == [0, 0, 0, 0] and qualities.max(axis=0).tolist() == [1, 1, 1, 1]


def test_one_member_alone_ranks_as_that_member() -> None:
    kernel = resolve_kernel("numpy")
    infeasible = np.any(ENSEMBLE_G > 0.0, axis=1)
    # CDP alone: every feasible row before every infeasible one.
    cdp = _ensemble((0, 1, 0, 0)).ranks(kernel, ENSEMBLE_F, ENSEMBLE_G, rng)
    assert cdp[~infeasible].max() < cdp[infeasible].min()
    # MCR alone: MCR's own order.
    mcr = _ensemble((0, 0, 1, 0)).ranks(kernel, ENSEMBLE_F, ENSEMBLE_G, rng)
    assert mcr.tolist() == MultipleConstraintRanking().ranks(kernel, ENSEMBLE_F, ENSEMBLE_G, rng).tolist()
    # NDS alone ignores the constraints: the Pareto fronts of the objectives come in order.
    nds = _ensemble((1, 0, 0, 0)).ranks(kernel, ENSEMBLE_F, ENSEMBLE_G, rng)
    assert np.all(np.diff(pareto_ranks(kernel, ENSEMBLE_F)[np.argsort(nds)]) >= 0)


def test_the_weights_are_normalized_to_sum_one() -> None:
    kernel = resolve_kernel("numpy")

    assert Ensemble((2, 2, 2, 2)).weights == (0.25, 0.25, 0.25, 0.25)
    assert _ensemble((2, 2, 2, 2)).ranks(kernel, ENSEMBLE_F, ENSEMBLE_G, rng).tolist() == (
        _ensemble((0.25, 0.25, 0.25, 0.25)).ranks(kernel, ENSEMBLE_F, ENSEMBLE_G, rng).tolist()
    )


def test_the_dynamic_penalty_grows_with_the_generation_and_is_the_state() -> None:
    kernel = resolve_kernel("numpy")
    early, late = _ensemble((0, 0, 0, 1), 0.01), _ensemble((0, 0, 0, 1), 0.9)

    assert early.state() == {"generation": 1.0} and late.state() == {"generation": pytest.approx(90.0)}
    assert not np.array_equal(early.qualities(kernel, ENSEMBLE_F, ENSEMBLE_G)[:, 3], late.qualities(kernel, ENSEMBLE_F, ENSEMBLE_G)[:, 3])
    restored = Ensemble()
    restored.restore(late.state())
    assert restored.penalty.generation == late.penalty.generation


# ---- Validation and tuning spaces -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (("ensemble", {"nds_weight": 0.0, "cdp_weight": 0.0, "mcr_weight": 0.0, "dpf_weight": 0.0}), "one must be above 0"),
        (("ensemble", {"mcr_weight": -1.0}), "mcr_weight must lie in"),
        (("ensemble", {"nds_weight": math.inf}), "they are finite"),
        (("ensemble", {"weights": [1, 1, 1, 1]}), "unknown: \\['weights'\\]"),
        (("multiple_constraint_ranking", {"nds_weight": 1.0}), "unknown: \\['nds_weight'\\]"),
    ],
)
def test_invalid_parameters_are_rejected(value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_constraint_handling(value)


def test_the_ensemble_takes_the_given_weights_and_defaults_the_others() -> None:
    technique = make_constraint_handling(("ensemble", {"mcr_weight": 0.5, "dpf_weight": 0.0}))

    assert isinstance(technique, Ensemble) and technique.total_order
    assert technique.weights == pytest.approx((0.25, 0.25, 0.5, 0.0))
    assert isinstance(make_constraint_handling("multiple_constraint_ranking"), MultipleConstraintRanking)


def test_large_finite_weights_normalize_without_overflow() -> None:
    # 1e308 + 1e308 overflows to infinity: the weights are scaled by the largest before they are summed.
    technique = make_constraint_handling(("ensemble", {"nds_weight": 1e308, "cdp_weight": 1e308, "mcr_weight": 0.0, "dpf_weight": 0.0}))

    assert isinstance(technique, Ensemble) and technique.weights == pytest.approx((0.5, 0.5, 0.0, 0.0))
    with pytest.raises(ValueError, match="finite weight"):
        Ensemble([math.nan, 1.0, 1.0, 1.0])


def test_only_the_nsgaii_space_offers_mcr_and_the_ensemble() -> None:
    space = builtin_parameter_space("nsgaii_constrained")
    technique = space.parameter("constraint_handling")
    weights = ["ensemble_nds_weight", "ensemble_cdp_weight", "ensemble_mcr_weight", "ensemble_dpf_weight"]
    assert technique.production("multiple_constraint_ranking") == ()
    assert [child.name for child in technique.production("ensemble")] == weights
    assert all((space.parameter(name).range, space.parameter(name).default) == ((0.0, 1.0), 0.25) for name in weights)

    defaults = dict(space.default_values())
    ensemble = {**defaults, "constraint_handling": "ensemble", **dict(zip(weights, (0.1, 0.2, 0.3, 0.4)))}
    expected = {"nds_weight": 0.1, "cdp_weight": 0.2, "mcr_weight": 0.3, "dpf_weight": 0.4}
    assert config_from_assignment("nsgaii_constrained", ensemble).constraint_handling == ("ensemble", expected)
    mcr = {**defaults, "constraint_handling": "multiple_constraint_ranking"}
    assert config_from_assignment("nsgaii_constrained", mcr).constraint_handling == ("multiple_constraint_ranking", {})

    for name in ("nsgaiii_constrained", "smsemoa_constrained", "agemoea_constrained"):
        values = builtin_parameter_space(name).parameter("constraint_handling").values
        assert "multiple_constraint_ranking" not in values and "ensemble" not in values
        with pytest.raises(ValueError, match="defined for NSGA-II only"):
            config_from_assignment(name, {**dict(builtin_parameter_space(name).default_values()), **mcr})
