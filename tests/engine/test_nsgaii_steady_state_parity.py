"""Steady-state NSGA-II follows the same trajectory under the NumPy and Numba engines.

With the Numba engine a supported steady-state configuration runs the compiled
one-offspring step; every other configuration keeps the reference ask/tell path.
Both must reproduce the NumPy engine exactly: the same populations at every
generation, the same final arrays and the same generator states.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

pytest.importorskip("numba")

import vamos
import vamos.engine.algorithm.nsgaii.ask_tell as ask_tell_module
import vamos.engine.algorithm.nsgaii.steady_state as steady_state_module
from vamos.algorithms import NSGAIIConfig
from vamos.engine.algorithm.nsgaii import NSGAII
from vamos.engine.archive import ExternalArchiveConfig
from vamos.engine.hooks.live_viz import NoOpLiveVisualization
from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.foundation.kernel.registry import resolve_kernel
from vamos.problems import DTLZ2, ZDT1

EVALUATIONS = 2000
PROBLEMS = {
    "zdt1": lambda: ZDT1(n_var=30),
    "dtlz2": lambda: DTLZ2(n_var=12, n_obj=3),
    "c1dtlz1": lambda: vamos.make_problem_selection("c1dtlz1", n_obj=3).instantiate(),
}


class _Trace(NoOpLiveVisualization):
    def __init__(self) -> None:
        self.frames: list[tuple[int, bytes, bytes]] = []

    def on_generation(self, generation: int, F: Any = None, X: Any = None, stats: Any = None) -> None:
        self.frames.append((generation, np.asarray(F).tobytes(), np.asarray(X).tobytes()))


def _compiled_step_available() -> bool:
    return NumbaKernel().nsga2_steady_state_kernels() is not None


def _run(problem: str, engine: str, seed: int, monkeypatch: pytest.MonkeyPatch, **changes: Any) -> dict[str, Any]:
    generators: list[np.random.Generator] = []
    original = np.random.default_rng

    def tracked_rng(*args: Any, **kwargs: Any) -> np.random.Generator:
        rng = original(*args, **kwargs)
        generators.append(rng)
        return rng

    instance = PROBLEMS[problem]()
    config = replace(NSGAIIConfig.default(pop_size=100, n_var=instance.n_var), steady_state=True, **changes)
    trace = _Trace()
    with monkeypatch.context() as patch:
        patch.setattr(np.random, "default_rng", tracked_rng)
        result = vamos.optimize(
            instance,
            algorithm="nsgaii",
            algorithm_config=config,
            engine=engine,
            seed=seed,
            max_evaluations=EVALUATIONS,
            live_viz=trace,
        )
    return {
        "evaluations": result.data["evaluations"],
        "X": np.asarray(result.X).tobytes(),
        "F": np.asarray(result.F).tobytes(),
        "population": tuple(np.asarray(result.data["population"][key]).tobytes() for key in ("X", "F")),
        "frames": trace.frames,
        "states": [deepcopy(rng.bit_generator.state) for rng in generators],
    }


@pytest.mark.numba
@pytest.mark.parametrize("seed", [5, 11])
@pytest.mark.parametrize("problem", sorted(PROBLEMS))
def test_steady_state_engines_follow_the_same_trajectory(problem: str, seed: int, monkeypatch: pytest.MonkeyPatch) -> None:
    reference = _run(problem, "numpy", seed, monkeypatch)
    compiled = _run(problem, "numba", seed, monkeypatch)

    assert reference["evaluations"] == compiled["evaluations"] == EVALUATIONS
    assert reference["states"] and compiled["states"] == reference["states"]
    assert len(compiled["frames"]) == len(reference["frames"]) == (EVALUATIONS - 100) // 100 + 1
    assert compiled["frames"] == reference["frames"]
    for key in ("X", "F", "population"):
        assert compiled[key] == reference[key]


@pytest.mark.numba
def test_steady_state_parity_with_an_external_archive(monkeypatch: pytest.MonkeyPatch) -> None:
    archive = ExternalArchiveConfig(capacity=40, pruning="crowding")
    reference = _run("zdt1", "numpy", 3, monkeypatch, external_archive=archive)
    compiled = _run("zdt1", "numba", 3, monkeypatch, external_archive=archive)

    assert compiled == reference


@pytest.mark.numba
def test_compiled_step_runs_only_for_the_numba_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    if not _compiled_step_available():
        pytest.skip("compiled pow differs from np.power on this platform; the reference path stays active")
    calls = {"ask": 0, "tell": 0}
    original_ask = steady_state_module.ask_steady_state
    original_tell = steady_state_module.tell_steady_state

    def counting_ask(*args: Any) -> Any:
        produced = original_ask(*args)
        calls["ask"] += produced is not None
        return produced

    def counting_tell(*args: Any) -> Any:
        replaced = original_tell(*args)
        calls["tell"] += replaced is not None
        return replaced

    monkeypatch.setattr(ask_tell_module, "ask_steady_state", counting_ask)
    monkeypatch.setattr(ask_tell_module, "tell_steady_state", counting_tell)
    config = replace(NSGAIIConfig.default(pop_size=20, n_var=6), steady_state=True)

    vamos.optimize(ZDT1(n_var=6), algorithm="nsgaii", algorithm_config=config, engine="numpy", seed=2, max_evaluations=60)
    assert calls == {"ask": 0, "tell": 0}

    result = vamos.optimize(ZDT1(n_var=6), algorithm="nsgaii", algorithm_config=config, engine="numba", seed=2, max_evaluations=60)
    assert result.data["evaluations"] == 60
    assert calls == {"ask": 40, "tell": 40}


@pytest.mark.numba
@pytest.mark.parametrize(
    "changes",
    [
        {"repair": ("reflect", {})},
        {"mutation": ("gaussian", {"prob": 0.2, "sigma": 0.1})},
        {"selection": ("tournament", {"size": 3})},
        {"track_genealogy": True},
    ],
    ids=["repair", "mutation", "selection", "genealogy"],
)
def test_unsupported_configurations_keep_the_reference_path(changes: dict[str, Any]) -> None:
    config = replace(NSGAIIConfig.default(pop_size=12, n_var=4), steady_state=True, **changes).to_dict()
    algorithm = NSGAII(config, kernel=NumbaKernel())
    algorithm._initialize_run(ZDT1(n_var=4), termination=("max_evaluations", 30), seed=0, eval_strategy=None, live_viz=None)
    assert algorithm._st is not None
    assert algorithm._st.steady_step is None


@pytest.mark.numba
def test_reference_steps_between_compiled_steps_keep_the_trajectory() -> None:
    """A step that leaves the compiled path (here a parent filter) rebuilds the cached fronts."""
    if not _compiled_step_available():
        pytest.skip("compiled pow differs from np.power on this platform; the reference path stays active")
    config = replace(NSGAIIConfig.default(pop_size=16, n_var=12), steady_state=True).to_dict()
    problem = DTLZ2(n_var=12, n_obj=3)
    runs = []
    for kernel_name in ("numpy", "numba"):
        algorithm = NSGAII(dict(config), kernel=resolve_kernel(kernel_name))
        algorithm._initialize_run(problem, termination=("max_evaluations", 400), seed=9, eval_strategy=None, live_viz=None)
        st = algorithm._st
        assert st is not None
        for step in range(300):
            # Every fifth step restricts the parents, which the compiled step does not support.
            st.parent_selection_filter = (lambda _state, ranks, _crowding: ranks <= 1) if step % 5 == 4 else None
            offspring = algorithm.ask()
            F = np.empty((1, 3))
            problem.evaluate(offspring, {"F": F})
            algorithm.tell(SimpleNamespace(F=F, G=None))
        runs.append((st.X.tobytes(), st.F.tobytes(), np.asarray(st.ranks).tobytes(), deepcopy(st.rng.bit_generator.state)))
    assert runs[0] == runs[1]
