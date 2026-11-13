"""Constraint-handling techniques in NSGA-III, SMS-EMOA and AGE-MOEA (see test_constraint_handling_techniques)."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np
import pytest

from vamos import optimize
from vamos.algorithms import AGEMOEAConfig, NSGAIIConfig, NSGAIIIConfig, SMSEMOAConfig
from vamos.engine.algorithm.agemoea.survival import technique_age_survival
from vamos.engine.algorithm.components.constraint_handling import CONSTRAINT_HANDLING_TECHNIQUES, ConstraintHandlingTechnique
from vamos.engine.algorithm.nsgaiii.helpers import nsgaiii_survival
from vamos.engine.algorithm.nsgaiii.nsgaiii import NSGAIII
from vamos.engine.algorithm.smsemoa.helpers import survival_selection
from vamos.engine.algorithm.smsemoa.smsemoa import SMSEMOA
from vamos.engine.algorithm.smsemoa.state import SMSEMOAState
from vamos.engine.tuning.racing.bridge import (
    build_agemoea_constrained_config_space,
    build_nsgaiii_constrained_config_space,
    build_smsemoa_constrained_config_space,
)
from vamos.engine.tuning.racing.bridge_assignments import config_from_assignment
from vamos.foundation.eval.backends import SerialEvalBackend
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.problems import LIRCMOP

PARAMETERS: dict[str, dict[str, float]] = {
    "cdp": {},
    "violated_constraints": {},
    "epsilon_constrained": {"initial_quantile": 0.2, "exponent": 5.0, "decay_fraction": 0.8},
    "push_pull": {
        "change_threshold": 1e-3,
        "window_fraction": 0.05,
        "decay_fraction": 0.8,
        "exponent": 2.0,
        "feasibility_threshold": 0.95,
        "reduction_rate": 0.1,
    },
    "stochastic_ranking": {"probability": 0.45},
    "self_adaptive_penalty": {},
}
ALGORITHMS = ["nsgaiii", "smsemoa", "agemoea"]
# SMS-EMOA ranks once per offspring, so it runs fewer evaluations.
EVALUATIONS = {"nsgaiii": 1500, "smsemoa": 500, "agemoea": 1500}


def _config(algorithm: str, technique: str | None, *, pop_size: int = 30) -> Any:
    if algorithm == "nsgaiii":
        config: Any = NSGAIIIConfig.default(pop_size=pop_size, n_obj=2)
    elif algorithm == "smsemoa":
        config = SMSEMOAConfig.default(pop_size=pop_size, n_var=10)
    else:
        config = AGEMOEAConfig.default(pop_size=pop_size, n_var=10)
    if technique is None:
        return config
    return dataclasses.replace(config, constraint_handling=(technique, PARAMETERS[technique]))


def _run(algorithm: str, technique: str | None, engine: str = "numpy", *, seed: int = 4) -> Any:
    return optimize(
        LIRCMOP(2, n_var=10),
        algorithm=algorithm,
        algorithm_config=_config(algorithm, technique),
        max_evaluations=EVALUATIONS[algorithm],
        seed=seed,
        engine=engine,
    )


class _FixedRanks(ConstraintHandlingTechnique):
    """A technique with fixed fronts, which records the progress of each update."""

    name = "fixed"

    def __init__(self, ranks: list[int]) -> None:
        self.fixed = np.asarray(ranks, dtype=np.int64)
        self.progress: list[float] = []

    def update(self, kernel: Any, F: Any, G: Any, progress: float) -> None:
        self.progress.append(progress)

    def ranks(self, kernel: Any, F: Any, G: Any, rng: np.random.Generator) -> Any:
        return self.fixed.copy()


# ---- Configuration -----------------------------------------------------------


@pytest.mark.parametrize("config_class", [NSGAIIConfig, NSGAIIIConfig, SMSEMOAConfig, AGEMOEAConfig])
def test_constraint_handling_is_the_last_optional_field(config_class: Any) -> None:
    fields = dataclasses.fields(config_class)

    assert fields[-1].name == "constraint_handling"
    assert fields[-1].default is None


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_the_builder_sets_the_technique_and_it_round_trips_through_json(algorithm: str) -> None:
    config_class = type(_config(algorithm, None))
    config = _config(algorithm, "push_pull")

    restored = config_class.from_dict(json.loads(json.dumps(config.to_dict())))

    assert restored == config
    assert restored.constraint_handling == ("push_pull", PARAMETERS["push_pull"])
    built = dataclasses.replace(_config(algorithm, None), constraint_handling=None)
    assert built.constraint_handling is None


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_the_builder_method_validates_the_technique(algorithm: str) -> None:
    config_class = type(_config(algorithm, None))
    builder = config_class.builder()

    assert builder.constraint_handling("stochastic_ranking", probability=0.3) is builder
    assert builder._cfg["constraint_handling"] == ("stochastic_ranking", {"probability": 0.3})
    with pytest.raises(ValueError, match="takes probability"):
        builder.constraint_handling("stochastic_ranking")


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_a_technique_needs_the_feasibility_mode(algorithm: str) -> None:
    config = dataclasses.replace(_config(algorithm, None), constraint_mode="none")

    with pytest.raises(ValueError, match="needs constraint_mode 'feasibility'"):
        dataclasses.replace(config, constraint_handling=("epsilon_constrained", PARAMETERS["epsilon_constrained"]))
    assert dataclasses.replace(config, constraint_handling="cdp").constraint_handling == ("cdp", {})


# ---- Survival over the technique's fronts ------------------------------------


def test_nsgaiii_keeps_whole_fronts_of_the_technique_and_niches_the_last() -> None:
    rng = np.random.default_rng(1)
    F = rng.random((8, 2))
    G = np.zeros((8, 1))
    # Front 0 is rows 2 and 4; front 1, rows 0, 1, 6 and 7, does not fit.
    technique = _FixedRanks([1, 1, 0, 2, 0, 2, 1, 1])

    survived = nsgaiii_survival(
        F[:4] * 0 + np.arange(4)[:, None],
        F[:4],
        G[:4],
        F[4:] * 0 + np.arange(4, 8)[:, None],
        F[4:],
        G[4:],
        4,
        np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
        rng,
        np.full(2, np.inf),
        None,
        np.full(2, -np.inf),
        kernel=resolve_kernel("numpy"),
        technique=technique,
        progress=0.25,
    )

    survivors = set(survived[3].tolist())
    assert {2, 4} <= survivors
    assert survivors - {2, 4} <= {0, 1, 6, 7}
    assert len(survivors) == 4
    assert technique.progress == [0.25]


def test_smsemoa_removes_from_the_last_front_of_the_technique() -> None:
    F = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    # The technique puts the first row, which no row dominates, in the last front.
    technique = _FixedRanks([1, 0, 0, 0])
    st = SMSEMOAState(
        X=np.arange(6, dtype=float).reshape(3, 2),
        F=F.copy(),
        G=np.zeros((3, 1)),
        rng=np.random.default_rng(0),
        constraint_mode="feasibility",
        constraint_handling=technique,
        ref_point=np.array([2.0, 2.0]),
        ref_adaptive=False,
    )

    survival_selection(st, np.array([[9.0, 9.0]]), np.array([[0.2, 0.9]]), np.zeros((1, 1)), resolve_kernel("numpy"), 0.5)

    assert st.F.tolist() == [[0.5, 0.5], [1.0, 0.0], [0.2, 0.9]]
    assert technique.progress == [0.5]
    assert st.ranks is None


def test_agemoea_keeps_whole_fronts_of_the_technique() -> None:
    rng = np.random.default_rng(2)
    F = rng.random((8, 2))
    technique = _FixedRanks([1, 0, 1, 2, 0, 2, 0, 1])

    survivors, scores = technique_age_survival(F, np.zeros((8, 1)), 5, resolve_kernel("numpy"), technique, rng, 0.1)

    assert {1, 4, 6} <= set(survivors.tolist())
    assert set(survivors.tolist()) - {1, 4, 6} <= {0, 2, 7}
    assert survivors.size == scores.size == 5
    assert technique.progress == [0.1]


# ---- Runs ----------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_explicit_cdp_reproduces_the_default(algorithm: str) -> None:
    default, explicit = _run(algorithm, None), _run(algorithm, "cdp")

    assert np.array_equal(default.X, explicit.X)
    assert np.array_equal(default.F, explicit.F)


@pytest.mark.parametrize("technique", CONSTRAINT_HANDLING_TECHNIQUES[1:])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_each_technique_runs_reproducibly(algorithm: str, technique: str) -> None:
    first, second = _run(algorithm, technique), _run(algorithm, technique)

    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.F, second.F)


@pytest.mark.parametrize("technique", CONSTRAINT_HANDLING_TECHNIQUES[1:])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_engines_give_the_same_run(algorithm: str, technique: str) -> None:
    pytest.importorskip("numba")
    reference, compiled = _run(algorithm, technique, "numpy"), _run(algorithm, technique, "numba")

    assert np.array_equal(reference.X, compiled.X)
    assert np.array_equal(reference.F, compiled.F)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_epsilon_changes_the_search(algorithm: str) -> None:
    assert not np.array_equal(_run(algorithm, "cdp").X, _run(algorithm, "epsilon_constrained").X)


@pytest.mark.parametrize(("algorithm", "algorithm_class"), [("nsgaiii", NSGAIII), ("smsemoa", SMSEMOA)])
def test_ask_and_tell_follow_the_progress_of_run(algorithm: str, algorithm_class: Any) -> None:
    """run() counts the offspring before the survival and tell() after it; both rank at the same progress."""
    problem, termination = LIRCMOP(2, n_var=10), ("max_evaluations", EVALUATIONS[algorithm])
    config = _config(algorithm, "epsilon_constrained").to_dict()
    kernel = resolve_kernel("numpy")
    whole = algorithm_class(config, kernel).run(problem, termination, seed=5)

    stepped = algorithm_class(config, kernel)
    stepped.initialize(problem, termination, seed=5)
    backend = SerialEvalBackend()
    while not stepped.should_terminate():
        X = stepped.ask()
        stepped.tell(backend.evaluate(X, problem))
    result = stepped.result()

    assert np.array_equal(whole["X"], result["X"])
    assert np.array_equal(whole["F"], result["F"])


# ---- Tuning spaces -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("space_name", "build_space"),
    [
        ("nsgaiii_constrained", build_nsgaiii_constrained_config_space),
        ("smsemoa_constrained", build_smsemoa_constrained_config_space),
        ("agemoea_constrained", build_agemoea_constrained_config_space),
    ],
)
def test_the_constrained_spaces_tune_the_technique(space_name: str, build_space: Any) -> None:
    space = build_space()
    names = {parameter.name for parameter in space.flatten({"constraint_handling": "epsilon_constrained"})}
    assert {"constraint_handling", "epsilon_initial_quantile", "epsilon_exponent", "epsilon_decay_fraction"} <= names

    assignment = space.sample(np.random.default_rng(0))
    assignment.update(
        constraint_handling="epsilon_constrained",
        epsilon_initial_quantile=0.3,
        epsilon_exponent=2.0,
        epsilon_decay_fraction=0.6,
    )
    config = config_from_assignment(space_name, assignment)

    assert config.constraint_handling == ("epsilon_constrained", {"initial_quantile": 0.3, "exponent": 2.0, "decay_fraction": 0.6})
