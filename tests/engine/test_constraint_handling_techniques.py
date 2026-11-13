"""Constraint-handling techniques: jMetal's ranks of fixed populations, and NSGA-II runs with each one."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vamos import optimize
from vamos.algorithms import NSGAIIConfig
from vamos.engine.algorithm.components.constrained_ranking import stochastic_ranking_order
from vamos.engine.algorithm.components.constraint_handling import (
    CONSTRAINT_HANDLING_TECHNIQUES,
    EpsilonConstrained,
    PushPullSearch,
    make_constraint_handling,
    normalize_constraint_handling,
)
from vamos.engine.algorithm.nsgaii.helpers import feasible_nsga2_survival
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.problems import LIRCMOP

FIXTURE = Path(__file__).resolve().parent / "data" / "constraint_handling_jmetal_ranks.json"
CASES: list[dict[str, Any]] = json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]
ENGINES = ["numpy", "numba"]

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


def _kernel(engine: str) -> Any:
    if engine == "numba":
        pytest.importorskip("numba")
    return resolve_kernel(engine)


def _config(technique: str | None, *, pop_size: int = 40) -> NSGAIIConfig:
    builder = (
        NSGAIIConfig.builder()
        .pop_size(pop_size)
        .crossover("sbx", prob=0.9, eta=20.0)
        .mutation("pm", prob="1/n", eta=20.0)
        .selection("tournament", size=2)
    )
    if technique is not None:
        builder.constraint_handling(technique, **PARAMETERS[technique])
    return builder.build()


def _run(technique: str | None, engine: str = "numpy", *, evaluations: int = 2000, seed: int = 4, **kwargs: Any) -> Any:
    return optimize(
        LIRCMOP(2, n_var=10),
        algorithm="nsgaii",
        algorithm_config=_config(technique),
        max_evaluations=evaluations,
        seed=seed,
        engine=engine,
        **kwargs,
    )


# ---- Ranks of fixed populations ---------------------------------------------


def _case_id(case: dict[str, Any]) -> str:
    return f"{case['technique']}-{CASES.index(case)}"


def test_the_fixture_covers_every_technique() -> None:
    assert {case["technique"] for case in CASES} == set(CONSTRAINT_HANDLING_TECHNIQUES)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_ranks_match_jmetal(case: dict[str, Any], engine: str) -> None:
    kernel = _kernel(engine)
    technique = make_constraint_handling((case["technique"], case["params"]))
    for step in case["updates"]:
        technique.update(kernel, np.array(step["F"]), np.array(step["G"]), step["progress"])
    rng = np.random.default_rng(case.get("uniform_seed", 0))

    ranks = technique.ranks(kernel, np.array(case["F"]), np.array(case["G"]), rng)

    assert ranks.tolist() == case["ranks"]
    if isinstance(technique, (EpsilonConstrained, PushPullSearch)):
        assert technique.epsilon == pytest.approx(case["epsilon"], rel=1e-12, abs=0.0)
    if isinstance(technique, PushPullSearch):
        assert technique.pushing == case["pushing"]


def test_compiled_stochastic_ranking_matches_the_reference() -> None:
    kernels = _kernel("numba").selection_kernels()
    rng = np.random.default_rng(11)
    for n in (1, 2, 17, 60):
        pareto_rank = rng.integers(0, 4, size=n).astype(np.int64)
        feasible = rng.random(n) < 0.4
        violation = np.where(feasible, 0.0, rng.random(n).round(2))
        uniforms = rng.random(n * max(0, n - 1))
        for probability in (0.0, 0.45, 1.0):
            expected = stochastic_ranking_order(pareto_rank, feasible, violation, uniforms, probability)
            got = kernels.stochastic_ranking_order(pareto_rank, feasible, violation, uniforms, probability)
            assert got.tolist() == expected.tolist()


@pytest.mark.parametrize("engine", ENGINES)
def test_cdp_ranks_survive_as_nsgaii_feasibility_survival(engine: str) -> None:
    # Without ties in the violation, the ranks of cdp keep the rows of NSGA-II's own cdp survival.
    kernel = _kernel(engine)
    rng = np.random.default_rng(3)
    for _ in range(20):
        F = rng.random((40, 2))
        G = np.where(rng.random((40, 2)) < 0.5, -rng.random((40, 2)), rng.random((40, 2)))
        ranks = make_constraint_handling("cdp").ranks(kernel, F, G, rng)
        _, _, _, kept = feasible_nsga2_survival(kernel, F[:20], F[:20], G[:20], F[20:], F[20:], G[20:], 25, return_indices=True)
        cut = np.sort(ranks)[24]
        assert set(np.flatnonzero(ranks < cut)) <= set(kept.tolist()) <= set(np.flatnonzero(ranks <= cut))


# ---- Validation -------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("lagrangian", "Unknown constraint-handling technique"),
        (("epsilon_constrained", {"initial_quantile": 0.5, "exponent": 2.0}), "missing: \\['decay_fraction'\\]"),
        (("cdp", {"exponent": 2.0}), "unknown: \\['exponent'\\]"),
        (("stochastic_ranking", {"probability": 1.5}), "probability must lie in"),
        (("epsilon_constrained", {"initial_quantile": 0.5, "exponent": 0.0, "decay_fraction": 0.5}), "exponent must lie in"),
        (("push_pull", {**PARAMETERS["push_pull"], "decay_fraction": 0.0}), "decay_fraction must lie in"),
        (3, "technique name or"),
    ],
)
def test_invalid_techniques_are_rejected(value: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_constraint_handling(value)


def test_none_is_cdp() -> None:
    assert normalize_constraint_handling(None) == ("cdp", {})
    assert _config(None).constraint_handling is None


def test_the_config_round_trips_through_json() -> None:
    config = _config("push_pull")

    restored = NSGAIIConfig.from_dict(json.loads(json.dumps(config.to_dict())))

    assert restored == config
    assert restored.constraint_handling == ("push_pull", PARAMETERS["push_pull"])


def test_a_technique_needs_the_feasibility_mode() -> None:
    with pytest.raises(ValueError, match="needs constraint_mode 'feasibility'"):
        NSGAIIConfig.builder().pop_size(10).crossover("sbx", prob=0.9, eta=20.0).mutation("pm", prob=0.1, eta=20.0).constraint_mode(
            "none"
        ).constraint_handling("stochastic_ranking", probability=0.4).build()


def test_a_changing_technique_needs_a_budget() -> None:
    with pytest.raises(ValueError, match="needs its progress"):
        EpsilonConstrained(0.5, 2.0, 0.5).update(resolve_kernel("numpy"), np.zeros((2, 2)), np.ones((2, 1)), float("nan"))


# ---- NSGA-II ----------------------------------------------------------------


def test_explicit_cdp_reproduces_the_default() -> None:
    default, explicit = _run(None), _run("cdp")

    assert np.array_equal(default.X, explicit.X)
    assert np.array_equal(default.F, explicit.F)


@pytest.mark.parametrize("technique", CONSTRAINT_HANDLING_TECHNIQUES[1:])
def test_each_technique_runs_reproducibly(technique: str) -> None:
    first, second = _run(technique), _run(technique)

    assert np.array_equal(first.X, second.X)
    assert np.array_equal(first.F, second.F)
    assert first.data["G"].shape == (first.F.shape[0], 2)


@pytest.mark.parametrize("technique", CONSTRAINT_HANDLING_TECHNIQUES[1:])
def test_engines_give_the_same_run(technique: str) -> None:
    pytest.importorskip("numba")
    reference, compiled = _run(technique, "numpy"), _run(technique, "numba")

    assert np.array_equal(reference.X, compiled.X)
    assert np.array_equal(reference.F, compiled.F)


def test_epsilon_changes_the_search() -> None:
    assert not np.array_equal(_run("cdp").X, _run("epsilon_constrained").X)


@pytest.mark.parametrize(("technique", "generation"), [("epsilon_constrained", 20), ("push_pull", 30)])
def test_a_resumed_run_keeps_the_technique_state(technique: str, generation: int, tmp_path: Path) -> None:
    # Restarting the technique would change epsilon, or push and pull search's window of ideal and nadir points.
    from vamos.engine.algorithm.nsgaii import NSGAII
    from vamos.foundation.checkpoint import load_checkpoint

    def run(**kwargs: Any) -> dict[str, Any]:
        algorithm = NSGAII(_config(technique).to_dict(), kernel=resolve_kernel("numpy"))
        return dict(algorithm.run(LIRCMOP(2, n_var=10), ("max_evaluations", 2000), seed=9, **kwargs))

    whole = run(checkpoint_dir=str(tmp_path), checkpoint_interval=1)
    checkpoint = load_checkpoint(tmp_path / f"nsgaii_seed9_gen{generation}.ckpt", trusted=True)
    assert "constraint_handling" in checkpoint["extra"]

    resumed = run(checkpoint=checkpoint)

    assert np.array_equal(whole["X"], resumed["X"])


class _StopAtGeneration:
    """A live callback that asks the run to stop once it reaches ``generation``."""

    def __init__(self, generation: int) -> None:
        self.generation = generation
        self.reached = False

    def on_start(self, ctx: object = None) -> None:
        return None

    def on_generation(self, generation: int, F: object = None, X: object = None, stats: object = None) -> None:
        self.reached = generation >= self.generation

    def on_end(self, final_F: object = None, final_stats: object = None) -> None:
        return None

    def should_stop(self) -> bool:
        return self.reached


@pytest.mark.parametrize(("technique", "generation"), [("epsilon_constrained", 20), ("push_pull", 30)])
def test_a_run_resumed_from_the_returned_checkpoint_keeps_the_technique_state(technique: str, generation: int) -> None:
    # The checkpoint in the result is the in-memory way to resume, without a checkpoint directory.
    from vamos.engine.algorithm.nsgaii import NSGAII

    def run(**kwargs: Any) -> dict[str, Any]:
        algorithm = NSGAII(_config(technique).to_dict(), kernel=resolve_kernel("numpy"))
        return dict(algorithm.run(LIRCMOP(2, n_var=10), ("max_evaluations", 2000), seed=9, **kwargs))

    whole = run()
    stopped = run(live_viz=_StopAtGeneration(generation))
    checkpoint = stopped["checkpoint"]
    assert checkpoint["generation"] == generation
    assert "constraint_handling" in checkpoint["extra"]

    resumed = run(checkpoint=checkpoint)

    assert np.array_equal(whole["X"], resumed["X"])
