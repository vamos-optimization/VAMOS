"""MOEA/D's parity with Evolver: jMetal's modified Tchebycheff, its DE base vector, and the exchanged names.

- ``inverse_weight_tchebycheff`` is jMetal 7.8's ``ModifiedTschebyscheff``, which Evolver calls
  ``modifiedTschebyscheff``: ``max_i |f_i - z*_i| / w_i``, a zero weight counting ``1e-7 |f_i - z*_i|``.
  Up to a positive factor per weight vector, it is the Tchebycheff function with the weights that
  Qi et al. (2014, Eq. 3) call the WS-transformation, ``(1 / w_i) / sum_j (1 / w_j)``. VAMOS's
  ``modified_tchebycheff`` keeps its meaning (the augmented formula with ``rho = 1e-3``), which Evolver
  does not have.
- ``base="random"`` makes the base vector of the DE mutant a random parent, distinct from those of the
  difference vectors, as Evolver's MOEA/D does (``populationAndNeighborhoodMatingPoolSelection`` passes
  ``selectCurrentSolution = false``, so the mating pool holds three, or five, random parents and jMetal's
  ``DifferentialEvolutionCrossover`` takes the last as the base); the child still starts as the current
  solution. ``base="current"`` (the default) is MOEA/D-DE's and jMetal's ``MOEADDEBuilder``'s.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.engine.algorithm.config import MOEADConfig
from vamos.engine.algorithm.moead import MOEAD
from vamos.engine.algorithm.moead.aggregation import build_aggregator, resolve_aggregation_spec, tchebycheff
from vamos.engine.algorithm.moead.helpers import ZERO_WEIGHT_EPS, update_neighborhood
from vamos.engine.algorithm.moead.state import MOEADState
from vamos.engine.operators.policies.moead import build_variation_operators, one_offspring_real_variation
from vamos.engine.tuning import config_from_assignment
from vamos.engine.tuning.racing.parameter_names import from_evolver, to_evolver
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem

NAME = "inverse_weight_tchebycheff"


# ---- jMetal 7.8's ModifiedTschebyscheff.compute(), transcribed ------------------------------------


def jmetal_modified_tschebyscheff(f, w, z, nad, normalize=False, eps=1e-6):  # type: ignore[no-untyped-def]
    max_fun = -1.0e30
    for n in range(len(f)):
        if normalize:
            diff = abs((f[n] - z[n]) / (nad[n] - z[n] + eps))
        else:
            diff = abs(f[n] - z[n])
        feval = 0.0000001 * diff if w[n] == 0 else diff / w[n]
        max_fun = max(max_fun, feval)
    return max_fun


def test_inverse_weight_tchebycheff_by_hand() -> None:
    aggregate = build_aggregator(NAME, {})
    # |f - z*| = [0.5, 1.0] divided by w = [0.25, 0.75]: [2, 4/3].
    assert aggregate(np.array([[0.5, 2.0]]), np.array([[0.25, 0.75]]), np.array([0.0, 1.0]))[0] == pytest.approx(2.0, abs=1e-15)
    # A zero weight counts its difference times 1e-7, as jMetal does.
    assert aggregate(np.array([[0.5, 2.0]]), np.array([[0.0, 1.0]]), np.array([0.0, 1.0]))[0] == pytest.approx(1.0, abs=1e-15)
    assert aggregate(np.array([[0.5, 1.0]]), np.array([[0.0, 1.0]]), np.array([0.0, 1.0]))[0] == pytest.approx(0.5e-7, abs=1e-22)


@pytest.mark.parametrize("params", [{}, {"normalize": True}, {"normalize": True, "epsilon": 0.25}])
def test_inverse_weight_tchebycheff_follows_jmetal(params: dict[str, Any]) -> None:
    rng = np.random.default_rng(5)
    for n_obj in (2, 3, 5):
        F = rng.uniform(0.0, 4.0, size=(25, n_obj))
        W = rng.dirichlet(np.ones(n_obj), size=25)
        W[0, 1] = 0.0
        ideal, nadir = F.min(axis=0) - 0.1, F.max(axis=0) + 0.2
        values = build_aggregator(NAME, params)(F, W, ideal, nadir)
        normalize, eps = bool(params.get("normalize", False)), float(params.get("epsilon", 1e-6))
        expected = [jmetal_modified_tschebyscheff(F[i], W[i], ideal, nadir, normalize, eps) for i in range(25)]
        np.testing.assert_allclose(values, expected, rtol=1e-14, atol=0.0)


def test_inverse_weight_tchebycheff_is_tchebycheff_with_the_ws_transformed_weights() -> None:
    # Qi et al. (2014), Eq. 3: WS(w)_i = (1 / w_i) / sum_j (1 / w_j); the factor sum_j (1 / w_j) is constant per subproblem.
    rng = np.random.default_rng(9)
    F = rng.uniform(0.0, 3.0, size=(40, 3))
    W = rng.dirichlet(np.ones(3), size=40)
    ideal = F.min(axis=0) - 0.05
    inverse = 1.0 / W
    ws = inverse / inverse.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(build_aggregator(NAME, {})(F, W, ideal), inverse.sum(axis=1) * tchebycheff(F, ws, ideal), rtol=1e-12)


def test_the_name_its_id_and_its_parameters() -> None:
    from vamos.foundation.kernel.moead_kernels import AGG_INVERSE_WEIGHT_TCHEBYCHEFF

    assert resolve_aggregation_spec(NAME, {}) == (AGG_INVERSE_WEIGHT_TCHEBYCHEFF, 5.0, 0.001)
    cfg = MOEADConfig.default(pop_size=20, n_var=6, n_obj=2)
    assert dataclasses.replace(cfg, aggregation=(NAME, {"normalize": True, "epsilon": 0.5})).aggregation[0] == NAME
    with pytest.raises(ConfigurationError, match=f"aggregation '{NAME}' takes normalize, epsilon; got 'rho'"):
        dataclasses.replace(cfg, aggregation=(NAME, {"rho": 0.1}))
    # modified_tchebycheff keeps its meaning: the augmented formula with rho = 1e-3.
    F, W, z = np.array([[0.5, 2.0]]), np.array([[0.25, 0.75]]), np.array([0.0, 1.0])
    assert build_aggregator("modified_tchebycheff", {})(F, W, z)[0] == pytest.approx(0.75 + 0.001 * 0.875)


def _state(params: dict[str, Any], rng: np.random.Generator, n_obj: int) -> MOEADState:
    pop = 12
    weights = rng.dirichlet(np.ones(n_obj), size=pop)
    weights[0, 1] = 0.0
    weights[0] /= weights[0].sum()
    agg_id, theta, rho = resolve_aggregation_spec(NAME, params)
    F = rng.uniform(0.0, 3.0, size=(pop, n_obj))
    return MOEADState(
        X=np.arange(pop, dtype=float)[:, None],
        F=F,
        G=None,
        rng=rng,
        pop_size=pop,
        offspring_size=1,
        constraint_mode="none",
        weights=weights,
        weights_safe=np.where(weights == 0.0, ZERO_WEIGHT_EPS, weights),
        weights_unit=weights / np.linalg.norm(weights, axis=1)[:, None],
        neighbors=np.tile(np.arange(pop), (pop, 1)),
        ideal=F.min(axis=0) - 0.05,
        aggregator=build_aggregator(NAME, params),
        aggregation_id=agg_id,
        aggregation_theta=theta,
        aggregation_rho=rho,
        aggregation_normalize=bool(params.get("normalize", False)),
        aggregation_epsilon=float(params.get("epsilon", 1e-6)),
        nadir=F.max(axis=0) + 0.3,
        replace_limit=4,
        batch_size=1,
    )


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "python"])
@pytest.mark.parametrize("params", [{}, {"normalize": True, "epsilon": 0.01}])
def test_the_neighbourhood_update_replaces_by_jmetals_values(
    params: dict[str, Any], compiled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vamos.engine.algorithm.moead import neighborhood

    if compiled:
        pytest.importorskip("numba")
        assert neighborhood.get_update_neighborhood_numba() is not None
    else:
        monkeypatch.setattr(neighborhood, "get_update_neighborhood_numba", lambda: None)
    rng = np.random.default_rng(13)
    normalize, eps = bool(params.get("normalize", False)), float(params.get("epsilon", 1e-6))
    for n_obj in (2, 3):
        for _ in range(20):
            state = _state(params, rng, n_obj)
            assert state.nadir is not None
            child_f = rng.uniform(0.0, 3.0, size=n_obj)
            order = rng.permutation(state.pop_size)
            expected_F = state.F.copy()
            replaced = 0
            for k in order:
                child = jmetal_modified_tschebyscheff(child_f, state.weights[k], state.ideal, state.nadir, normalize, eps)
                current = jmetal_modified_tschebyscheff(expected_F[k], state.weights[k], state.ideal, state.nadir, normalize, eps)
                if child < current:
                    expected_F[k] = child_f
                    replaced += 1
                    if replaced >= state.replace_limit:
                        break
            update_neighborhood(
                st=state, idx=0, child=np.array([99.0]), child_f=child_f, child_g=None, cv_penalty=0.0, candidate_order=order
            )
            np.testing.assert_array_equal(state.F, expected_F)


def _config(aggregation: tuple[str, dict[str, Any]] = (NAME, {}), *, batch_size: int = 1, **crossover: Any) -> MOEADConfig:
    return (
        MOEADConfig.builder()
        .pop_size(20)
        .batch_size(batch_size)
        .neighbor_size(6)
        .delta(0.9)
        .replace_limit(2)
        .crossover("de", cr=0.8, f=0.5, **crossover)
        .mutation("pm", prob="1/n", eta=20.0)
        .aggregation(aggregation[0], **aggregation[1])
        .weight_vectors(divisions=19)
        .build()
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("params", [{}, {"normalize": True}])
def test_the_engines_give_the_same_run(params: dict[str, Any], batch_size: int) -> None:
    pytest.importorskip("numba")
    from vamos.foundation.kernel.numba_backend import NumbaKernel

    cfg = _config((NAME, params), batch_size=batch_size).to_dict()
    problem = ZDT1Problem(n_var=8)
    expected = MOEAD(dict(cfg), kernel=NumPyKernel()).run(problem, ("max_evaluations", 603), seed=5)
    algorithm = MOEAD(dict(cfg), kernel=NumbaKernel())
    actual = algorithm.run(problem, ("max_evaluations", 603), seed=5)
    for key in ("X", "F"):
        assert actual["population"][key].tobytes() == expected["population"][key].tobytes()
    assert actual["checkpoint"]["rng_state"] == expected["checkpoint"]["rng_state"]
    # The compiled one-offspring step computes the function itself; with normalization it needs the nadir point.
    if batch_size == 1 and NumbaKernel().variation_kernels() is not None:
        assert (algorithm._steady is None) == bool(params.get("normalize"))


def test_the_function_changes_the_search() -> None:
    problem = ZDT1Problem(n_var=8)
    runs = {
        name: MOEAD(_config((name, {})).to_dict(), kernel=NumPyKernel()).run(problem, ("max_evaluations", 603), seed=5)
        for name in (NAME, "tchebycheff", "modified_tchebycheff")
    }
    assert runs[NAME]["evaluations"] == 603
    assert runs[NAME]["population"]["F"].tobytes() != runs["tchebycheff"]["population"]["F"].tobytes()
    assert runs[NAME]["population"]["F"].tobytes() != runs["modified_tchebycheff"]["population"]["F"].tobytes()


# ---- The names exchanged with Evolver -------------------------------------------------------------


def test_evolvers_tchebycheff_variants_read_into_their_vamos_functions() -> None:
    assert from_evolver({"aggregationFunction": "modifiedTschebyscheff"}) == {"aggregation_function": NAME}
    assert from_evolver({"aggregationFunction": "augmentedTschebyscheff"}) == {"aggregation_function": "augmented_tchebycheff"}
    # Evolver's MOEADDoubleDefault.txt with the aggregation changed.
    evolver_default = {
        "neighborhoodSize": 20,
        "maximumNumberOfReplacedSolutions": 2,
        "createInitialSolutions": "default",
        "crossoverProbability": 0.9,
        "mutation": "polynomial",
        "mutationProbabilityFactor": 1.0,
        "polynomialMutationDistributionIndex": 20.0,
        "crossover": "SBX",
        "sbxDistributionIndex": 20.0,
        "neighborhoodSelectionProbability": 0.9,
    }
    for evolver, name in (("modifiedTschebyscheff", NAME), ("augmentedTschebyscheff", "augmented_tchebycheff")):
        assignment = {"pop_size": 100, **from_evolver({**evolver_default, "aggregationFunction": evolver})}
        config = config_from_assignment("moead", assignment)
        assert isinstance(config, MOEADConfig) and config.aggregation == (name, {})


def test_the_vamos_functions_write_evolvers_names() -> None:
    assert to_evolver({"aggregation": NAME}) == {"aggregationFunction": "modifiedTschebyscheff"}
    assert to_evolver({"aggregation_function": "augmented_tchebycheff"}) == {"aggregationFunction": "augmentedTschebyscheff"}
    assert to_evolver({"aggregation": "tchebycheff"}) == {"aggregationFunction": "tschebyscheff"}


@pytest.mark.parametrize("key", ["aggregation", "aggregation_function"])
def test_vamos_modified_tchebycheff_has_no_evolver_name(key: str) -> None:
    with pytest.raises(ValueError, match="modified_tchebycheff has no Evolver equivalent"):
        to_evolver({key: "modified_tchebycheff", "neighbor_size": 20})


# ---- The base vector of the DE mutant -------------------------------------------------------------


class _ScriptedRng:
    """Return scripted arrays from ``random`` and ``integers``, in call order."""

    def __init__(self, *arrays: Any) -> None:
        self._arrays = [np.asarray(a) for a in arrays]

    def random(self, size: Any = None) -> np.ndarray:
        return self._arrays.pop(0).astype(float).copy()

    def integers(self, low: int, high: int | None = None, size: Any = None) -> np.ndarray:
        return self._arrays.pop(0).astype(np.int64).copy()


def _crossover(scheme: str, base: str | None, *, cr: float, f: float, n_var: int) -> Any:
    params: dict[str, Any] = {"cr": cr, "f": f, "scheme": scheme}
    if base is not None:
        params["base"] = base
    cfg = {"crossover": ("de", params), "mutation": ("pm", {"prob": 0.1, "eta": 20.0})}
    xl, xu = np.full(n_var, -100.0), np.full(n_var, 100.0)
    return build_variation_operators(cfg, "real", n_var, xl, xu, np.random.default_rng(0))[0]


def test_a_random_base_by_hand() -> None:
    r0, r1, rb, current = np.array([3.0, 3.0, 3.0]), np.array([1.0, 1.0, 1.0]), np.array([10.0, 20.0, 30.0]), np.zeros(3)
    parents = np.stack([r0, r1, rb, current])[None]
    # Mutant: rb + 0.5 (r0 - r1) = [11, 21, 31]; variable 0 keeps the current solution (0.9 >= 0.5), 2 is jrand.
    child = _crossover("rand_1_bin", "random", cr=0.5, f=0.5, n_var=3)(parents, _ScriptedRng([[0.9, 0.1, 0.7]], [2]))[0, 0]
    np.testing.assert_array_equal(child, [0.0, 21.0, 31.0])
    r2, r3 = np.array([2.0, 2.0, 2.0]), np.zeros(3)
    parents = np.stack([r0, r1, r2, r3, rb, current])[None]
    # rand/2: rb + 0.5 (r0 - r1) + 0.5 (r2 - r3) = [12, 22, 32]; the block starts at 1 and takes 1 and 2.
    child = _crossover("rand_2_exp", "random", cr=0.5, f=0.5, n_var=3)(parents, _ScriptedRng([1], [[0.2, 0.7, 0.0]]))[0, 0]
    np.testing.assert_array_equal(child, [0.0, 22.0, 32.0])


def _jmetal_child(scheme: str, parents: np.ndarray, f: float, cr: float, j: int, draws: np.ndarray, jrand: int) -> np.ndarray:
    """``DifferentialEvolutionCrossover.execute`` (jMetal 7.8) for one child; ``parents[-1]`` is the current solution."""
    n = parents.shape[1]
    child = parents[-1].copy()

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


@pytest.mark.parametrize("cr", [0.0, 0.4, 1.0])
@pytest.mark.parametrize("scheme", ["rand_1_bin", "rand_1_exp", "rand_2_bin", "rand_2_exp"])
def test_the_children_with_a_random_base_follow_jmetal(scheme: str, cr: float) -> None:
    # Evolver's mating pool, (r0, r1, r2) or (r0, ..., r4), then the current solution that the crossover copies.
    width = 4 if scheme.startswith("rand_1") else 6
    parents = np.random.default_rng(21).uniform(-1.0, 1.0, size=(30, width, 6))
    children = _crossover(scheme, "random", cr=cr, f=0.7, n_var=6)(parents, np.random.default_rng(4))[:, 0, :]
    twin = np.random.default_rng(4)
    if scheme.endswith("bin"):
        draws, j = twin.random((30, 6)), twin.integers(0, 6, size=30)
        expected = [_jmetal_child(scheme, parents[i], 0.7, cr, 0, draws[i], int(j[i])) for i in range(30)]
    else:
        start = twin.integers(0, 6, size=30)
        draws = twin.random((30, 6))
        expected = [_jmetal_child(scheme, parents[i], 0.7, cr, int(start[i]), draws[i], -1) for i in range(30)]
    np.testing.assert_array_equal(children, np.asarray(expected))


def test_the_current_base_is_the_default() -> None:
    parents = np.random.default_rng(1).uniform(size=(9, 3, 6))
    default = _crossover("rand_1_bin", None, cr=0.4, f=0.7, n_var=6)(parents, np.random.default_rng(5))
    explicit = _crossover("rand_1_bin", "current", cr=0.4, f=0.7, n_var=6)(parents, np.random.default_rng(5))
    assert explicit.tobytes() == default.tobytes()
    problem = ZDT1Problem(n_var=8)
    plain = MOEAD(_config(("pbi", {})).to_dict(), kernel=NumPyKernel()).run(problem, ("max_evaluations", 403), seed=2)
    current = MOEAD(_config(("pbi", {}), base="current").to_dict(), kernel=NumPyKernel()).run(problem, ("max_evaluations", 403), seed=2)
    assert current["population"]["F"].tobytes() == plain["population"]["F"].tobytes()


@pytest.mark.parametrize(("scheme", "count"), [("rand_1_bin", 3), ("rand_2_exp", 5)])
@pytest.mark.parametrize("delta", [1.0, 0.0])
@pytest.mark.parametrize("batch_size", [1, 6])
def test_a_random_base_draws_one_more_distinct_parent(scheme: str, count: int, delta: float, batch_size: int) -> None:
    cfg = dataclasses.replace(_config(("tchebycheff", {}), scheme=scheme, base="random", batch_size=batch_size), delta=delta)
    algorithm = MOEAD(cfg.to_dict(), kernel=NumPyKernel())
    algorithm.initialize(ZDT1Problem(n_var=8), ("max_evaluations", 400), seed=4)
    state = algorithm._st
    assert state is not None
    for _ in range(5):
        algorithm.ask()
        parents, active = state.pending_parent_pairs, state.pending_active_indices
        assert parents is not None and active is not None
        assert parents.shape == (batch_size, count)
        for row, index in zip(parents, active, strict=True):
            assert len(set(row.tolist())) == count
            pool = state.neighbors[index] if delta == 1.0 else np.arange(state.pop_size)
            assert set(row.tolist()) <= set(pool.tolist())


def test_the_random_base_is_uniform_over_the_neighbourhood() -> None:
    cfg = dataclasses.replace(_config(("tchebycheff", {}), base="random", batch_size=20), delta=1.0, neighbor_size=5)
    algorithm = MOEAD(cfg.to_dict(), kernel=NumPyKernel())
    algorithm.initialize(ZDT1Problem(n_var=8), ("max_evaluations", 10_000), seed=4)
    state = algorithm._st
    assert state is not None
    counts = np.zeros((3, 5))
    for _ in range(250):
        algorithm.ask()
        assert state.pending_parent_pairs is not None and state.pending_active_indices is not None
        for row, index in zip(state.pending_parent_pairs, state.pending_active_indices, strict=True):
            for slot, parent in enumerate(row):
                counts[slot, list(state.neighbors[index]).index(parent)] += 1
    np.testing.assert_allclose(counts / 5000.0, 0.2, atol=0.025)


@pytest.mark.parametrize("scheme", ["rand_1_bin", "rand_2_bin"])
def test_the_engines_agree_with_a_random_base(scheme: str) -> None:
    pytest.importorskip("numba")
    from vamos.foundation.kernel.numba_backend import NumbaKernel

    cfg = _config(("tchebycheff", {}), scheme=scheme, base="random")
    operators = build_variation_operators(
        cfg.to_dict(), "real", 8, np.zeros(8), np.ones(8), np.random.default_rng(0), variation_kernels=NumbaKernel().variation_kernels()
    )
    assert one_offspring_real_variation(*operators, 8) is None
    problem = ZDT1Problem(n_var=8)
    expected = MOEAD(cfg.to_dict(), kernel=NumPyKernel()).run(problem, ("max_evaluations", 603), seed=6)
    algorithm = MOEAD(cfg.to_dict(), kernel=NumbaKernel())
    actual = algorithm.run(problem, ("max_evaluations", 603), seed=6)
    assert algorithm._steady is None
    assert actual["population"]["F"].tobytes() == expected["population"]["F"].tobytes()
    current = MOEAD(_config(("tchebycheff", {}), scheme=scheme).to_dict(), kernel=NumPyKernel()).run(
        problem, ("max_evaluations", 603), seed=6
    )
    assert current["population"]["F"].tobytes() != expected["population"]["F"].tobytes()


def test_genealogy_records_the_random_base() -> None:
    cfg = _config(("tchebycheff", {}), base="random", batch_size=3).to_dict()
    cfg["track_genealogy"] = True
    algorithm = MOEAD(cfg, kernel=NumPyKernel())
    algorithm.initialize(ZDT1Problem(n_var=8), ("max_evaluations", 60), seed=6)
    state = algorithm._st
    assert state is not None and state.genealogy_tracker is not None and state.ids is not None
    algorithm.ask()
    assert state.pending_offspring_ids is not None and state.pending_parent_pairs is not None
    for child, parents, index in zip(state.pending_offspring_ids, state.pending_parent_pairs, state.pending_active_indices, strict=True):
        expected = [int(state.ids[p]) for p in (*parents, index)]
        assert len(expected) == 4
        assert state.genealogy_tracker.records[int(child)].parents == expected


def test_the_options_round_trip_and_replay_exactly(tmp_path: Any) -> None:
    cfg = _config((NAME, {"normalize": True}), scheme="rand_2_bin", base="random")
    assert cfg.crossover == ("de", {"cr": 0.8, "f": 0.5, "scheme": "rand_2_bin", "base": "random"})
    assert MOEADConfig.from_dict(cfg.to_dict()) == cfg
    result = vamos.optimize("zdt1", n_var=8, algorithm="moead", algorithm_config=cfg, max_evaluations=120, seed=2, engine="numpy")
    stored = vamos.save_result(result, tmp_path / "source")
    resolved = vamos.load_run(stored.root).manifest.resolved_spec
    assert resolved["operators"]["crossover"]["config"]["base"] == "random"
    assert resolved["algorithm"]["config"]["aggregation"][0] == NAME
    assert vamos.verify_run(stored.root).effective_replayability == "exact"
    assert vamos.reproduce(stored.root, output=tmp_path / "replay").exact is True


@pytest.mark.parametrize(("scheme", "neighbor_size", "needed"), [("rand_1_bin", 2, 3), ("rand_2_exp", 4, 5)])
def test_a_random_base_needs_one_more_neighbour(scheme: str, neighbor_size: int, needed: int) -> None:
    message = f"{scheme} with base='random' needs neighbor_size >= {needed}"
    with pytest.raises(ConfigurationError, match=message):
        dataclasses.replace(_config(("tchebycheff", {}), scheme=scheme, base="random"), neighbor_size=neighbor_size)
    cfg = _config(("tchebycheff", {}), scheme=scheme).to_dict()
    cfg["neighbor_size"] = neighbor_size
    cfg["crossover"] = ("de", {"cr": 0.8, "f": 0.5, "scheme": scheme, "base": "random"})
    with pytest.raises(ConfigurationError, match=message):
        MOEAD(cfg, kernel=NumPyKernel()).run(ZDT1Problem(n_var=8), ("max_evaluations", 60), seed=1)


@pytest.mark.parametrize("base", ["best", "Random ", "", 0])
def test_an_unknown_base_is_rejected(base: object) -> None:
    if base == "Random ":
        assert _config(("tchebycheff", {}), base=base).crossover[1]["base"] == "random"
        return
    with pytest.raises(ConfigurationError, match="DE crossover base must be one of 'current', 'random'"):
        _config(("tchebycheff", {}), base=base)


def test_a_base_needs_the_de_crossover() -> None:
    builder = MOEADConfig.builder().pop_size(20).neighbor_size(6).delta(0.9).replace_limit(2).mutation("pm").aggregation("pbi")
    with pytest.raises(ConfigurationError, match="The base applies to the DE crossover"):
        builder.crossover("sbx", prob=0.9, eta=20.0, base="random").build()
