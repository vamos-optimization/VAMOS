"""How NSGA-II survives with a constraint-handling technique: the total-order flag and the ``survived`` hook.

A technique with ``total_order`` puts each row in a front of its own, so NSGA-II skips the crowding distance,
which cannot change a front of one row; these tests show that the runs are bit-identical. After each
replacement NSGA-II gives the technique the survivors (``survived``), which checkpoints keep through ``state``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.engine.algorithm.components.constrained_ranking import pareto_ranks
from vamos.engine.algorithm.components.constraint_handling import ConstraintHandlingTechnique, make_constraint_handling
from vamos.engine.algorithm.components.constraint_two_rankings import TwoRankings
from vamos.engine.algorithm.nsgaii import NSGAII, constrained_survival
from vamos.engine.algorithm.nsgaii.run import _checkpoint_extra
from vamos.foundation.eval.backends import SerialEvalBackend
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.problems import LIRCMOP

# Generational, batches of ten offspring, and steady state (one offspring per replacement).
SCHEDULES = {"generational": {}, "batches": {"offspring_size": 10}, "steady_state": {"steady_state": True}}
EVALUATIONS = {"generational": 2000, "batches": 1000, "steady_state": 400}


def _config(technique: tuple[str, dict[str, float]], **schedule: Any) -> NSGAIIConfig:
    config = (
        NSGAIIConfig.builder()
        .pop_size(30)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
        .build()
    )
    return dataclasses.replace(config, constraint_handling=technique, **schedule)


# The techniques with a total order, and their parameters.
TOTAL_ORDERS = {
    "two_rankings": {},
    "infeasibility_driven": {"proportion": 0.2},
    "adaptive_tradeoff": {},
    "multiple_constraint_ranking": {},
    "ensemble": {},
}


def _run(schedule: str, engine: str = "numpy", technique: str = "two_rankings") -> Any:
    return optimize(
        LIRCMOP(2, n_var=10),
        algorithm="nsgaii",
        algorithm_config=_config((technique, TOTAL_ORDERS[technique]), **SCHEDULES[schedule]),
        max_evaluations=EVALUATIONS[schedule],
        seed=7,
        engine=engine,
    )


@pytest.mark.parametrize("technique", list(TOTAL_ORDERS))
@pytest.mark.parametrize("engine", ["numpy", "numba"])
@pytest.mark.parametrize("schedule", list(SCHEDULES))
def test_skipping_crowding_for_a_total_order_is_bit_identical(
    schedule: str, engine: str, technique: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if engine == "numba":
        pytest.importorskip("numba")
    calls: list[int] = []
    crowding = constrained_survival.compute_crowding

    def counted(F: Any, fronts: Any) -> Any:
        calls.append(len(fronts))
        return crowding(F, fronts)

    monkeypatch.setattr(constrained_survival, "compute_crowding", counted)
    skipped = _run(schedule, engine, technique)
    assert calls == []

    monkeypatch.setattr(type(make_constraint_handling((technique, TOTAL_ORDERS[technique]))), "total_order", False)
    computed = _run(schedule, engine, technique)
    assert calls

    assert np.array_equal(skipped.X, computed.X)
    assert np.array_equal(skipped.F, computed.F)
    assert np.array_equal(skipped.data["G"], computed.data["G"])


class _Recorder(ConstraintHandlingTechnique):
    """Pareto ranks, recording each population that survives."""

    name = "recorder"

    def __init__(self) -> None:
        self.survivors: list[tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]] = []

    def ranks(self, kernel: Any, F: Any, G: Any, rng: np.random.Generator) -> Any:
        return pareto_ranks(kernel, F)

    def survived(self, F: Any, G: Any) -> None:
        self.survivors.append((np.array(F), np.array(G)))

    def state(self) -> dict[str, Any]:
        return {"replacements": len(self.survivors)}


@pytest.mark.parametrize("schedule", ["generational", "steady_state"])
def test_the_technique_learns_each_population_that_survives(schedule: str) -> None:
    problem = LIRCMOP(2, n_var=10)
    algorithm = NSGAII(_config(("violation_objective", {}), **SCHEDULES[schedule]).to_dict(), kernel=resolve_kernel("numpy"))
    algorithm.initialize(problem, ("max_evaluations", 300), seed=3)
    state = algorithm._st
    assert state is not None
    recorder = _Recorder()
    state.constraint_handling = recorder
    backend = SerialEvalBackend()

    replacements = 0
    while not algorithm.should_terminate():
        algorithm.tell(backend.evaluate(algorithm.ask(), problem))
        replacements += 1
        assert len(recorder.survivors) == replacements
        F, G = recorder.survivors[-1]
        assert np.array_equal(F, state.F)
        assert np.array_equal(G, state.G)

    assert replacements > 1
    assert _checkpoint_extra(state)["constraint_handling"] == {"replacements": replacements}


def test_the_default_hook_and_flag_change_nothing() -> None:
    technique = ConstraintHandlingTechnique()

    assert technique.total_order is False
    assert technique.survived(np.zeros((2, 2)), np.zeros((2, 1))) is None
    assert TwoRankings.total_order is True
    for technique, params in TOTAL_ORDERS.items():
        assert make_constraint_handling((technique, params)).total_order is True
