"""The DE schemes of MOEA/D's ``de`` crossover, against jMetal's ``DifferentialEvolutionCrossover``.

jMetal 7.8 (Evolver's ``differentialEvolutionCrossover``) defines ``RAND_1_BIN``, ``RAND_1_EXP``,
``RAND_2_BIN`` and ``RAND_2_EXP``. The child starts as a copy of the current solution (the
subproblem's); the mutant is ``parent[2] + F (parent[0] - parent[1])`` for rand/1 and
``parent[4] + F (parent[0] - parent[1]) + F (parent[2] - parent[3])`` for rand/2, where the last
parent is the current solution (jMetal's ``MOEADDEBuilder`` selects it); the binomial crossover
takes each variable from the mutant when its draw is below CR or it is ``jrand``, and the
exponential one takes the variable ``j`` (drawn) and the next ones, cyclically, while a draw is
below CR, up to all of them. The repair then clips the child to the bounds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import vamos
from vamos.engine.algorithm.config import MOEADConfig
from vamos.engine.algorithm.moead import MOEAD
from vamos.engine.operators.policies.moead import build_variation_operators, one_offspring_real_variation
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem

SCHEMES = ("rand_1_bin", "rand_1_exp", "rand_2_bin", "rand_2_exp")


class _ScriptedRng:
    """Return scripted arrays from ``random`` and ``integers``, in call order."""

    def __init__(self, *arrays: Any) -> None:
        self._arrays = [np.asarray(a) for a in arrays]

    def random(self, size: Any = None) -> np.ndarray:
        return self._arrays.pop(0).astype(float).copy()

    def integers(self, low: int, high: int | None = None, size: Any = None) -> np.ndarray:
        return self._arrays.pop(0).astype(np.int64).copy()


def _crossover(scheme: str | None, *, cr: float, f: float, n_var: int, lower: float = -100.0, upper: float = 100.0) -> Any:
    params: dict[str, Any] = {"cr": cr, "f": f}
    if scheme is not None:
        params["scheme"] = scheme
    cfg = {"crossover": ("de", params), "mutation": ("pm", {"prob": 0.1, "eta": 20.0})}
    xl, xu = np.full(n_var, lower), np.full(n_var, upper)
    crossover, _ = build_variation_operators(cfg, "real", n_var, xl, xu, np.random.default_rng(0))
    return crossover


# ---- Hand-computed children -------------------------------------------------------------


def test_rand_1_exp_takes_a_cyclic_block_from_the_mutant() -> None:
    base, p0, p1 = np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0]), np.zeros(4)
    parents = np.stack([np.stack([p0, p1, base])] * 3)
    # The mutant is 0 + 0.5 (p0 - p1) = [0.5, 1, 1.5, 2]. Row 0 starts at 2 and stops at its second
    # draw (0.9 >= 0.5); row 1 starts at 3 and wraps to 0 and 1; row 2 takes everything.
    start = np.array([2, 3, 1])
    draws = np.array([[0.1, 0.9, 0.0, 0.0], [0.1, 0.2, 0.9, 0.0], [0.0, 0.0, 0.0, 0.0]])
    children = _crossover("rand_1_exp", cr=0.5, f=0.5, n_var=4)(parents, _ScriptedRng(start, draws))[:, 0, :]
    np.testing.assert_array_equal(children[0], [0.0, 0.0, 1.5, 2.0])
    np.testing.assert_array_equal(children[1], [0.5, 1.0, 0.0, 2.0])
    np.testing.assert_array_equal(children[2], [0.5, 1.0, 1.5, 2.0])


def test_rand_1_exp_takes_one_variable_with_cr_zero() -> None:
    parents = np.stack([np.stack([np.ones(5), np.zeros(5), np.zeros(5)])])
    children = _crossover("rand_1_exp", cr=0.0, f=1.0, n_var=5)(parents, _ScriptedRng([4], np.zeros((1, 5))))[:, 0, :]
    np.testing.assert_array_equal(children[0], [0.0, 0.0, 0.0, 0.0, 1.0])


def test_rand_2_bin_adds_two_difference_vectors() -> None:
    base = np.array([1.0, 1.0, 1.0])
    p0, p1, p2, p3 = np.array([3.0, 5.0, 7.0]), np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0]), np.array([0.0, 4.0, 1.0])
    parents = np.stack([p0, p1, p2, p3, base])[None]
    # Mutant: 1 + 0.5 (p0 - p1) + 0.5 (p2 - p3) = [3, 2, 4.5]; variable 0 keeps the base (0.6 >= 0.5), 1 is jrand.
    children = _crossover("rand_2_bin", cr=0.5, f=0.5, n_var=3)(parents, _ScriptedRng([[0.6, 0.8, 0.3]], [1]))[:, 0, :]
    np.testing.assert_array_equal(children[0], [1.0, 2.0, 4.5])


def test_rand_2_exp_adds_two_difference_vectors() -> None:
    base = np.zeros(3)
    p0, p1, p2, p3 = np.array([2.0, 4.0, 6.0]), np.zeros(3), np.array([1.0, 1.0, 1.0]), np.zeros(3)
    parents = np.stack([p0, p1, p2, p3, base])[None]
    # Mutant: 0.5 p0 + 0.5 p2 = [1.5, 2.5, 3.5]; start at 1, take 1 and 2, stop at the second draw.
    children = _crossover("rand_2_exp", cr=0.5, f=0.5, n_var=3)(parents, _ScriptedRng([1], [[0.2, 0.7, 0.0]]))[:, 0, :]
    np.testing.assert_array_equal(children[0], [0.0, 2.5, 3.5])


def test_the_repair_clips_the_child() -> None:
    parents = np.stack([np.stack([np.full(2, 5.0), np.zeros(2), np.zeros(2)])])
    children = _crossover("rand_1_exp", cr=1.0, f=1.0, n_var=2, lower=0.0, upper=1.0)(parents, _ScriptedRng([0], np.zeros((1, 2))))
    np.testing.assert_array_equal(children[:, 0, :], [[1.0, 1.0]])


# ---- jMetal's loops, with the draws of the Generator ------------------------------------


def _jmetal_child(scheme: str, parents: np.ndarray, f: float, cr: float, j: int, draws: np.ndarray, jrand: int) -> np.ndarray:
    """``DifferentialEvolutionCrossover.execute`` (jMetal 7.8) for one child, transcribed."""
    n = parents.shape[1]
    current = parents[-1]
    child = current.copy()

    def mutate(index: int) -> float:
        if scheme.startswith("rand_1"):
            return float(parents[2][index] + f * (parents[0][index] - parents[1][index]))
        return float(parents[4][index] + f * (parents[0][index] - parents[1][index]) + f * (parents[2][index] - parents[3][index]))

    if scheme.endswith("bin"):
        for index in range(n):
            if draws[index] < cr or index == jrand:
                child[index] = mutate(index)
    else:
        length, position = 0, 0
        while True:
            child[j] = mutate(j)
            j = (j + 1) % n
            length += 1
            draw = draws[position]
            position += 1
            if not (draw < cr and length < n):
                break
    return child


@pytest.mark.parametrize("cr", [0.0, 0.3, 0.9, 1.0])
@pytest.mark.parametrize("scheme", SCHEMES)
def test_the_children_follow_jmetal(scheme: str, cr: float) -> None:
    n_parents = 3 if scheme.startswith("rand_1") else 5
    rng = np.random.default_rng(17)
    parents = rng.uniform(-1.0, 1.0, size=(40, n_parents, 7))
    f = 0.6
    children = _crossover(scheme, cr=cr, f=f, n_var=7)(parents, np.random.default_rng(3))[:, 0, :]

    twin = np.random.default_rng(3)
    if scheme.endswith("bin"):
        draws, j = twin.random((40, 7)), twin.integers(0, 7, size=40)
        expected = [_jmetal_child(scheme, parents[i], f, cr, 0, draws[i], int(j[i])) for i in range(40)]
    else:
        start = twin.integers(0, 7, size=40)
        draws = twin.random((40, 7))
        expected = [_jmetal_child(scheme, parents[i], f, cr, int(start[i]), draws[i], -1) for i in range(40)]
    np.testing.assert_array_equal(children, np.asarray(expected))


def test_rand_1_bin_is_the_operator_of_1_3_0() -> None:
    parents = np.random.default_rng(1).uniform(size=(9, 3, 6))
    default = _crossover(None, cr=0.4, f=0.7, n_var=6)(parents, np.random.default_rng(5))
    explicit = _crossover("rand_1_bin", cr=0.4, f=0.7, n_var=6)(parents, np.random.default_rng(5))
    assert explicit.tobytes() == default.tobytes()


# ---- MOEA/D with each scheme ---------------------------------------------------------------


def _moead(scheme: str, *, batch_size: int = 1, delta: float = 0.9, neighbor_size: int = 6) -> MOEADConfig:
    return (
        MOEADConfig.builder()
        .pop_size(20)
        .batch_size(batch_size)
        .neighbor_size(neighbor_size)
        .delta(delta)
        .replace_limit(2)
        .crossover("de", cr=0.8, f=0.5, scheme=scheme)
        .mutation("pm", prob="1/n", eta=20.0)
        .aggregation("tchebycheff")
        .weight_vectors(divisions=19)
        .build()
    )


@pytest.mark.parametrize("delta", [1.0, 0.0])
@pytest.mark.parametrize("batch_size", [1, 6])
def test_rand_2_draws_four_distinct_parents_from_the_mating_pool(batch_size: int, delta: float) -> None:
    algorithm = MOEAD(_moead("rand_2_bin", batch_size=batch_size, delta=delta).to_dict(), kernel=NumPyKernel())
    algorithm.initialize(ZDT1Problem(n_var=8), ("max_evaluations", 400), seed=4)
    state = algorithm._st
    assert state is not None
    for _ in range(5):
        algorithm.ask()
        parents, active = state.pending_parent_pairs, state.pending_active_indices
        assert parents is not None and active is not None
        assert parents.shape == (batch_size, 4)
        for row, index in zip(parents, active, strict=True):
            assert len(set(row.tolist())) == 4
            pool = state.neighbors[index] if delta == 1.0 else np.arange(state.pop_size)
            assert set(row.tolist()) <= set(pool.tolist())
        state.pending_offspring = None


def test_rand_2_draws_each_ordered_parent_uniformly() -> None:
    algorithm = MOEAD(_moead("rand_2_bin", batch_size=20, delta=1.0, neighbor_size=5).to_dict(), kernel=NumPyKernel())
    algorithm.initialize(ZDT1Problem(n_var=8), ("max_evaluations", 10_000), seed=4)
    state = algorithm._st
    assert state is not None
    counts = np.zeros((4, 5))
    for _ in range(250):
        algorithm.ask()
        assert state.pending_parent_pairs is not None and state.pending_active_indices is not None
        for row, index in zip(state.pending_parent_pairs, state.pending_active_indices, strict=True):
            for slot, parent in enumerate(row):
                counts[slot, list(state.neighbors[index]).index(parent)] += 1
    # 5000 children: each neighbour takes each of the four places a fifth of the time.
    np.testing.assert_allclose(counts / 5000.0, 0.2, atol=0.025)


@pytest.mark.parametrize("scheme", SCHEMES[1:])
def test_the_compiled_step_leaves_the_other_schemes_to_the_reference_step(scheme: str) -> None:
    pytest.importorskip("numba")
    from vamos.foundation.kernel.numba_backend import NumbaKernel

    cfg = _moead(scheme)
    xl, xu = np.zeros(8), np.ones(8)
    kernels = NumbaKernel().variation_kernels()
    operators = build_variation_operators(cfg.to_dict(), "real", 8, xl, xu, np.random.default_rng(0), variation_kernels=kernels)
    assert one_offspring_real_variation(*operators, 8) is None

    problem = ZDT1Problem(n_var=8)
    expected = MOEAD(cfg.to_dict(), kernel=NumPyKernel()).run(problem, ("max_evaluations", 603), seed=6)
    algorithm = MOEAD(cfg.to_dict(), kernel=NumbaKernel())
    actual = algorithm.run(problem, ("max_evaluations", 603), seed=6)
    assert algorithm._steady is None
    for key in ("X", "F"):
        assert actual["population"][key].tobytes() == expected["population"][key].tobytes()
    assert actual["checkpoint"]["rng_state"] == expected["checkpoint"]["rng_state"]


@pytest.mark.parametrize("scheme", SCHEMES)
def test_each_scheme_runs_reproducibly_and_differs_from_the_others(scheme: str) -> None:
    problem = ZDT1Problem(n_var=8)
    first = MOEAD(_moead(scheme).to_dict(), kernel=NumPyKernel()).run(problem, ("max_evaluations", 603), seed=6)
    second = MOEAD(_moead(scheme).to_dict(), kernel=NumPyKernel()).run(problem, ("max_evaluations", 603), seed=6)
    assert first["population"]["F"].tobytes() == second["population"]["F"].tobytes()
    assert first["evaluations"] == 603
    others = [
        MOEAD(_moead(other).to_dict(), kernel=NumPyKernel()).run(problem, ("max_evaluations", 603), seed=6)
        for other in SCHEMES
        if other != scheme
    ]
    assert all(other["population"]["F"].tobytes() != first["population"]["F"].tobytes() for other in others)


def test_genealogy_records_the_five_parents_of_rand_2() -> None:
    cfg = _moead("rand_2_exp", batch_size=3).to_dict()
    cfg["track_genealogy"] = True
    algorithm = MOEAD(cfg, kernel=NumPyKernel())
    algorithm.initialize(ZDT1Problem(n_var=8), ("max_evaluations", 60), seed=6)
    state = algorithm._st
    assert state is not None and state.genealogy_tracker is not None and state.ids is not None
    algorithm.ask()
    assert state.pending_offspring_ids is not None and state.pending_parent_pairs is not None
    for child, parents, index in zip(state.pending_offspring_ids, state.pending_parent_pairs, state.pending_active_indices, strict=True):
        expected = [int(state.ids[p]) for p in (*parents, index)]
        assert len(expected) == 5
        assert state.genealogy_tracker.records[int(child)].parents == expected


def test_a_scheme_round_trips_and_replays_exactly(tmp_path: Any) -> None:
    cfg = _moead("rand_2_exp")
    assert cfg.crossover == ("de", {"cr": 0.8, "f": 0.5, "scheme": "rand_2_exp"})
    assert MOEADConfig.from_dict(cfg.to_dict()) == cfg
    result = vamos.optimize("zdt1", n_var=8, algorithm="moead", algorithm_config=cfg, max_evaluations=120, seed=2, engine="numpy")
    stored = vamos.save_result(result, tmp_path / "source")
    assert vamos.load_run(stored.root).manifest.resolved_spec["operators"]["crossover"]["config"]["scheme"] == "rand_2_exp"
    assert vamos.verify_run(stored.root).effective_replayability == "exact"
    assert vamos.reproduce(stored.root, output=tmp_path / "replay").exact is True


# ---- Validation -----------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", ["rand/1/exp", "RAND_1_BIN ", "best_1_bin", "", 1])
def test_an_unknown_scheme_is_rejected(scheme: object) -> None:
    if isinstance(scheme, str) and scheme.strip().lower() in SCHEMES:
        assert _moead(scheme).crossover[1]["scheme"] == scheme.strip().lower()
        return
    with pytest.raises(ConfigurationError, match="scheme must be one of 'rand_1_bin', 'rand_1_exp', 'rand_2_bin', 'rand_2_exp'"):
        _moead(scheme)  # type: ignore[arg-type]


def test_a_scheme_needs_the_de_crossover() -> None:
    builder = MOEADConfig.builder().pop_size(20).neighbor_size(6).delta(0.9).replace_limit(2).mutation("pm").aggregation("pbi")
    with pytest.raises(ConfigurationError, match="scheme applies to the DE crossover"):
        builder.crossover("sbx", prob=0.9, eta=20.0, scheme="rand_1_exp").build()


def test_rand_2_needs_four_neighbours() -> None:
    with pytest.raises(ConfigurationError, match="rand_2_bin needs neighbor_size >= 4"):
        _moead("rand_2_bin", neighbor_size=3)
    cfg = _moead("rand_1_bin", neighbor_size=3).to_dict()
    cfg["crossover"] = ("de", {"cr": 0.8, "f": 0.5, "scheme": "rand_2_exp"})
    with pytest.raises(ConfigurationError, match="rand_2_exp needs neighbor_size >= 4"):
        MOEAD(cfg, kernel=NumPyKernel()).run(ZDT1Problem(n_var=8), ("max_evaluations", 60), seed=1)
