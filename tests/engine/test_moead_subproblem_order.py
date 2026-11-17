"""The order in which MOEA/D visits its subproblems: ``subproblem_order``.

``"random_permutation"`` (the default) visits a new random permutation of the subproblems
in each cycle, as jMetal's ``RandomPermutationCycle``; ``"cyclic"`` visits 0, 1, ...,
``pop_size - 1`` and starts again, as jMetal's ``CyclicIntegerSequence``, and draws no
random number for it. They are Evolver's ``subProblemIdGenerator`` values
``randomPermutationCycle`` and ``cyclicIntegerSequence``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from vamos.engine.algorithm.config import MOEADConfig
from vamos.engine.algorithm.moead import MOEAD
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem


def _config(order: str | None = "cyclic", *, batch_size: int = 1, pop_size: int = 12) -> MOEADConfig:
    builder = (
        MOEADConfig.builder()
        .pop_size(pop_size)
        .batch_size(batch_size)
        .neighbor_size(4)
        .delta(0.9)
        .replace_limit(2)
        .crossover("de", cr=1.0, f=0.5)
        .mutation("pm", prob="1/n", eta=20.0)
        .aggregation("tchebycheff")
        .weight_vectors(divisions=pop_size - 1)
    )
    if order is not None:
        builder.subproblem_order(order)
    return builder.build()


def _visited(cfg: MOEADConfig, steps: int, kernel: Any = None) -> list[int]:
    algorithm = MOEAD(cfg.to_dict(), kernel=kernel or NumPyKernel())
    problem = ZDT1Problem(n_var=6)
    algorithm.initialize(problem, ("max_evaluations", 10_000), seed=5)
    visited: list[int] = []
    for _ in range(steps):
        X = algorithm.ask()
        state = algorithm._st
        assert state is not None and state.pending_active_indices is not None
        visited.extend(int(i) for i in state.pending_active_indices)
        F = np.empty((X.shape[0], 2))
        out = {"F": F}
        problem.evaluate(X, out)
        from vamos.foundation.eval import EvaluationResult

        algorithm.tell(EvaluationResult(F=out["F"], G=None))
    return visited


def test_the_default_is_the_random_permutation() -> None:
    assert MOEADConfig.default(pop_size=12, n_var=6, n_obj=2).subproblem_order == "random_permutation"
    assert _config(None).subproblem_order == "random_permutation"
    visited = _visited(_config(None), 24)
    assert sorted(visited[:12]) == list(range(12)) and sorted(visited[12:]) == list(range(12))
    assert visited[:12] != list(range(12))


@pytest.mark.parametrize("batch_size", [1, 5])
def test_the_cyclic_order_visits_the_subproblems_in_index_order(batch_size: int) -> None:
    visited = _visited(_config("cyclic", batch_size=batch_size), 30 // batch_size)
    assert visited == [i % 12 for i in range(len(visited))]


def test_the_cyclic_order_draws_no_random_number() -> None:
    algorithm = MOEAD(_config("cyclic").to_dict(), kernel=NumPyKernel())
    algorithm.initialize(ZDT1Problem(n_var=6), ("max_evaluations", 100), seed=5)
    state = algorithm._st
    assert state is not None
    np.testing.assert_array_equal(state.subproblem_order, np.arange(12))
    # The random order draws its permutation after the initial population; the cyclic order does not draw it.
    random_order = MOEAD(_config("random_permutation").to_dict(), kernel=NumPyKernel())
    random_order.initialize(ZDT1Problem(n_var=6), ("max_evaluations", 100), seed=5)
    assert random_order._st is not None
    rng = np.random.default_rng()
    rng.bit_generator.state = state.rng.bit_generator.state
    rng.permutation(12)
    assert rng.bit_generator.state == random_order._st.rng.bit_generator.state


@pytest.mark.numba
def test_the_compiled_step_visits_the_cyclic_order_and_matches_the_numpy_engine() -> None:
    pytest.importorskip("numba")
    from vamos.foundation.kernel.numba_backend import NumbaKernel

    cfg = _config("cyclic", pop_size=30)
    problem = ZDT1Problem(n_var=6)
    expected = MOEAD(cfg.to_dict(), kernel=NumPyKernel()).run(problem, ("max_evaluations", 401), seed=2)
    algorithm = MOEAD(cfg.to_dict(), kernel=NumbaKernel())
    actual = algorithm.run(problem, ("max_evaluations", 401), seed=2)
    if NumbaKernel().variation_kernels() is not None:
        assert algorithm._steady is not None
    for key in ("X", "F"):
        assert actual["population"][key].tobytes() == expected["population"][key].tobytes()
    assert actual["checkpoint"]["rng_state"] == expected["checkpoint"]["rng_state"]
    np.testing.assert_array_equal(actual["checkpoint"]["extra"]["subproblem_order"], np.arange(30))
    # 401 evaluations: 30 initial and 371 steps, so the cursor stands at 371 mod 30.
    assert actual["checkpoint"]["extra"]["subproblem_cursor"] == expected["checkpoint"]["extra"]["subproblem_cursor"] == 371 % 30


class _StopAtGeneration:
    """A live callback that asks the run to stop after ``generation`` generations."""

    def __init__(self, generation: int) -> None:
        self.generation = generation
        self.reached = False

    def on_start(self, ctx: object = None) -> None:
        return None

    def on_generation(self, generation: int, F: object = None, X: object = None, stats: object = None) -> None:
        self.reached = self.reached or generation >= self.generation

    def on_end(self, final_F: object = None, final_stats: object = None) -> None:
        return None

    def should_stop(self) -> bool:
        return self.reached


@pytest.mark.parametrize("batch_size", [1, 4])
def test_a_resumed_cyclic_run_continues_the_uninterrupted_run(batch_size: int) -> None:
    cfg = _config("cyclic", batch_size=batch_size, pop_size=20).to_dict()
    problem = ZDT1Problem(n_var=6)

    def run(**kwargs: Any) -> dict[str, Any]:
        return MOEAD(dict(cfg), kernel=NumPyKernel()).run(problem, ("max_evaluations", 500), seed=8, **kwargs)

    whole = run()
    stopped = run(live_viz=_StopAtGeneration(7))
    checkpoint = stopped["checkpoint"]
    assert 20 < checkpoint["n_eval"] < 500
    np.testing.assert_array_equal(checkpoint["extra"]["subproblem_order"], np.arange(20))

    resumed = run(checkpoint=checkpoint)

    for key in ("X", "F"):
        assert resumed["population"][key].tobytes() == whole["population"][key].tobytes()
    assert resumed["checkpoint"]["rng_state"] == whole["checkpoint"]["rng_state"]
    assert resumed["checkpoint"]["extra"]["subproblem_cursor"] == whole["checkpoint"]["extra"]["subproblem_cursor"]


def test_a_cyclic_run_refuses_a_checkpoint_of_a_random_order() -> None:
    problem = ZDT1Problem(n_var=6)
    random_run = MOEAD(_config("random_permutation", pop_size=20).to_dict(), kernel=NumPyKernel())
    checkpoint = random_run.run(problem, ("max_evaluations", 50), seed=8)["checkpoint"]
    assert not np.array_equal(checkpoint["extra"]["subproblem_order"], np.arange(20))

    with pytest.raises(ConfigurationError, match="subproblem_order"):
        MOEAD(_config("cyclic", pop_size=20).to_dict(), kernel=NumPyKernel()).run(
            problem, ("max_evaluations", 100), seed=8, checkpoint=checkpoint
        )


def test_the_option_round_trips_and_reaches_the_configuration() -> None:
    cfg = _config("cyclic")
    payload = cfg.to_dict()
    assert payload["subproblem_order"] == "cyclic"
    assert MOEADConfig.from_dict(payload) == cfg
    assert dataclasses.replace(cfg, subproblem_order=" Cyclic ").subproblem_order == "cyclic"
    # Fields added after 1.0 come last, so that the constructor keeps the positions of its parameters.
    names = [field.name for field in dataclasses.fields(MOEADConfig)]
    assert names.index("subproblem_order") > names.index("live_callback_mode")


@pytest.mark.parametrize("value", ["random", "cyclicIntegerSequence", "", 3, None])
def test_an_unknown_order_is_rejected(value: object) -> None:
    with pytest.raises(ConfigurationError, match="subproblem_order must be one of 'random_permutation', 'cyclic'"):
        MOEADConfig.builder().subproblem_order(value)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="subproblem_order must be one of"):
        dataclasses.replace(_config(None), subproblem_order=value)  # type: ignore[arg-type]
    if value is not None:  # A configuration mapping without the option runs the default order.
        cfg = _config(None).to_dict()
        cfg["subproblem_order"] = value
        with pytest.raises(ConfigurationError, match="subproblem_order must be one of"):
            MOEAD(cfg, kernel=NumPyKernel()).run(ZDT1Problem(n_var=6), ("max_evaluations", 40), seed=1)


def test_a_cyclic_run_replays_exactly(tmp_path: Any) -> None:
    import vamos

    result = vamos.optimize(
        "zdt1", n_var=6, algorithm="moead", algorithm_config=_config("cyclic"), max_evaluations=60, seed=2, engine="numpy"
    )
    stored = vamos.save_result(result, tmp_path / "source")
    assert vamos.load_run(stored.root).manifest.resolved_spec["algorithm"]["config"]["subproblem_order"] == "cyclic"
    assert vamos.verify_run(stored.root).effective_replayability == "exact"
    assert vamos.reproduce(stored.root, output=tmp_path / "replay").exact is True
