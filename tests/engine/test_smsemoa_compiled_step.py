"""The compiled SMS-EMOA steady-state step reproduces the reference path bit for bit."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

import numpy as np
import pytest

pytest.importorskip("numba")

import vamos
from vamos.algorithms import SMSEMOAConfig
from vamos.engine.algorithm.smsemoa import compiled_step, smsemoa
from vamos.engine.hooks.live_viz import NoOpLiveVisualization
from vamos.foundation.kernel._numba_smsemoa_measures import front_contributions_2d_match_reference, violation_matches_reference
from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.problems import ZDT1

PROBLEMS = (
    ("zdt1", {}),
    ("dtlz2", {"n_obj": 3}),
    ("c1dtlz1", {"n_obj": 3}),
)


class _Trace(NoOpLiveVisualization):
    def __init__(self) -> None:
        self.frames: list[tuple[int, bytes]] = []

    def on_generation(self, generation, F=None, X=None, stats=None):  # noqa: ANN001 - live-viz protocol
        self.frames.append((generation, F.tobytes()))


class _StepCounter:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.steps = 0
        self.references = 0
        step, reference = compiled_step.CompiledSteadyStep.step, compiled_step.CompiledSteadyStep._reference

        def counted_step(inner: Any, *args: Any) -> None:
            self.steps += 1
            step(inner, *args)

        def counted_reference(inner: Any, *args: Any) -> None:
            self.references += 1
            reference(inner, *args)

        monkeypatch.setattr(compiled_step.CompiledSteadyStep, "step", counted_step)
        monkeypatch.setattr(compiled_step.CompiledSteadyStep, "_reference", counted_reference)


def _run(monkeypatch: pytest.MonkeyPatch, problem: str, engine: str, *, compiled: bool = True, **kwargs: Any):
    """Run once, recording every live frame and the final state of every run-owned generator."""
    generators: list[np.random.Generator] = []
    original = np.random.default_rng

    def tracked_rng(*args: Any, **rng_kwargs: Any) -> np.random.Generator:
        rng = original(*args, **rng_kwargs)
        generators.append(rng)
        return rng

    with monkeypatch.context() as patch:
        patch.setattr(np.random, "default_rng", tracked_rng)
        if not compiled:
            patch.setattr(smsemoa, "build_compiled_step", lambda st, kernel: None)
        trace = _Trace()
        result = vamos.optimize(problem, algorithm="smsemoa", engine=engine, live_viz=trace, **kwargs)
    states = [deepcopy(rng.bit_generator.state) for rng in generators]
    return result, trace.frames, states


def _assert_same_run(expected: tuple[Any, Any, Any], actual: tuple[Any, Any, Any]) -> None:
    (ref_result, ref_frames, ref_states), (result, frames, states) = expected, actual
    assert states == ref_states
    assert frames == ref_frames
    for name in ("X", "F"):
        assert getattr(result, name).tobytes() == getattr(ref_result, name).tobytes()
    for name in ("X", "F"):
        assert result.data["population"][name].tobytes() == ref_result.data["population"][name].tobytes()
    assert result.data["reference_point"].tobytes() == ref_result.data["reference_point"].tobytes()
    assert result.data["evaluations"] == ref_result.data["evaluations"]


@pytest.fixture(autouse=True)
def _warm_probes() -> None:
    # The one-time probes build their own generators; run them before any generator is tracked.
    front_contributions_2d_match_reference()
    for n_constraints in (1, 2):
        violation_matches_reference(n_constraints)


@pytest.mark.numba
@pytest.mark.parametrize("seed", (5, 11))
@pytest.mark.parametrize("problem,kwargs", PROBLEMS, ids=[name for name, _ in PROBLEMS])
def test_compiled_step_follows_the_numpy_trajectory(monkeypatch, problem, kwargs, seed) -> None:
    counter = _StepCounter(monkeypatch)
    numpy_run = _run(monkeypatch, problem, "numpy", seed=seed, max_evaluations=2000, **kwargs)
    assert counter.steps == 0
    numba_run = _run(monkeypatch, problem, "numba", seed=seed, max_evaluations=2000, **kwargs)
    _assert_same_run(numpy_run, numba_run)
    pop_size = numba_run[0].data["population"]["X"].shape[0]
    assert numba_run[0].data["evaluations"] == 2000
    assert counter.steps == 2000 - pop_size
    assert counter.references == 0


def _config(*changes: Callable[[Any], Any]) -> SMSEMOAConfig:
    builder = SMSEMOAConfig.builder().pop_size(40).crossover("sbx", prob=1.0, eta=20.0).mutation("pm", prob="1/n", eta=20.0)
    builder = builder.selection("random").reference_point(adaptive=True)
    for change in changes:
        builder = change(builder)
    return builder.build()


VARIANTS: dict[str, Callable[[Any], Any]] = {
    "tournament": lambda b: b.selection("tournament", size=2),
    "tournament3": lambda b: b.selection("tournament", size=3),
    "duplicates": lambda b: b.eliminate_duplicates(True),
    "fixed_reference": lambda b: b.reference_point(vector=[1.0e4] * 3, adaptive=False),
    "offset": lambda b: b.reference_point(offset=0.25, adaptive=True),
    "partial_crossover": lambda b: b.crossover("sbx", prob=0.7, eta=15.0).mutation("pm", prob=0.2, eta=10.0),
    "archive": lambda b: b.external_archive(30),
}


@pytest.mark.numba
@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("problem,kwargs", PROBLEMS, ids=[name for name, _ in PROBLEMS])
def test_compiled_step_matches_the_reference_step(monkeypatch, problem, kwargs, variant) -> None:
    config = _config(VARIANTS[variant])
    if variant == "fixed_reference" and problem == "zdt1":
        config = _config(lambda b: b.reference_point(vector=[1.0e4] * 2, adaptive=False))
    counter = _StepCounter(monkeypatch)
    run_kwargs = dict(seed=7, max_evaluations=1237, algorithm_config=config, **kwargs)
    reference = _run(monkeypatch, problem, "numba", compiled=False, **run_kwargs)
    assert counter.steps == 0
    actual = _run(monkeypatch, problem, "numba", **run_kwargs)
    _assert_same_run(reference, actual)
    assert actual[0].data["evaluations"] == 1237
    assert counter.steps == 1237 - 40


@pytest.mark.numba
def test_compiled_step_keeps_the_reference_caches(monkeypatch) -> None:
    """Ranks, crowding and front lists left in the state match the reference path."""
    states = []
    for compiled in (False, True):
        with monkeypatch.context() as patch:
            if not compiled:
                patch.setattr(smsemoa, "build_compiled_step", lambda st, kernel: None)
            algorithm = smsemoa.SMSEMOA(_config(VARIANTS["tournament"]).to_dict(), NumbaKernel())
            algorithm.run(ZDT1(n_var=8), ("max_evaluations", 600), seed=3)
        states.append(algorithm.state)
    reference, actual = states
    assert reference is not None and actual is not None
    assert actual.fronts == reference.fronts
    assert actual.ranks.tobytes() == reference.ranks.tobytes()
    assert actual.crowding.tobytes() == reference.crowding.tobytes()


@pytest.mark.numba
@pytest.mark.parametrize(
    "changes",
    [
        lambda b: b.track_genealogy(True),
        lambda b: b.repair("reflect"),
        lambda b: b.crossover("blx_alpha", prob=1.0),
        lambda b: b.mutation("gaussian", prob=0.2, sigma=0.05),
    ],
    ids=["genealogy", "reflect-repair", "blx-crossover", "gaussian-mutation"],
)
def test_unsupported_configurations_keep_the_reference_step(monkeypatch, changes) -> None:
    counter = _StepCounter(monkeypatch)
    config = _config(changes)
    result = vamos.optimize("zdt1", algorithm="smsemoa", algorithm_config=config, engine="numba", seed=2, max_evaluations=300)
    assert counter.steps == 0
    assert result.data["evaluations"] == 300


def test_numpy_engine_keeps_the_reference_step(monkeypatch) -> None:
    counter = _StepCounter(monkeypatch)
    vamos.optimize("zdt1", algorithm="smsemoa", engine="numpy", seed=2, max_evaluations=200)
    assert counter.steps == 0
