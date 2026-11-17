"""MOEA/D's augmented Tchebycheff and inverted PBI aggregations and the normalization of the objectives.

The reference formulas are transcriptions of jMetal 7.8's ``AugmentedTschebyscheff``,
``InvertedPenaltyBoundaryIntersection``, ``Tschebyscheff``, ``WeightedSum`` and
``PenaltyBoundaryIntersection`` (``org.uma.jmetal.util.aggregationfunction.impl``), which
Evolver's ``AggregationFunctionParameter`` builds. With normalization each objective becomes
``(f - z*) / (z^nad - z* + epsilon)`` (inverted PBI: ``(z^nad - f) / (z^nad - z* + epsilon)``);
jMetal's ``MOEADReplacement`` estimates ``z^nad`` as the largest value of each objective in the
non-dominated set of every solution evaluated so far (compared with jMetal's
``DominanceWithConstraintsComparator``), the new solution included.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.engine.algorithm.config import MOEADConfig
from vamos.engine.algorithm.moead import MOEAD
from vamos.engine.algorithm.moead.aggregation import (
    aggregation_methods,
    augmented_tchebycheff,
    build_aggregator,
    inverted_pbi,
    resolve_aggregation_spec,
)
from vamos.engine.algorithm.moead.helpers import ZERO_WEIGHT_EPS, update_neighborhood
from vamos.engine.algorithm.moead.nadir import add_to_front, nadir_point, nondominated_front
from vamos.engine.algorithm.moead.state import MOEADState
from vamos.foundation.exceptions import ConfigurationError
from vamos.foundation.kernel.moead_kernels import AGG_AUGMENTED_TCHEBYCHEFF, AGG_INVERTED_PBI
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.zdt1 import ZDT1Problem

# ---- jMetal 7.8's compute(), transcribed ---------------------------------------------------


def _value(f: np.ndarray, z: np.ndarray, nad: np.ndarray, n: int, normalize: bool, eps: float) -> float:
    return float((f[n] - z[n]) / (nad[n] - z[n] + eps)) if normalize else float(f[n] - z[n])


def jmetal_tschebyscheff(f, w, z, nad, normalize=False, eps=1e-6, rho=None):  # type: ignore[no-untyped-def]
    max_fun, total = -1.0e30, 0.0
    for n in range(len(f)):
        diff = abs(_value(f, z, nad, n, normalize, eps))
        feval = 0.0001 * diff if w[n] == 0 else diff * w[n]
        max_fun = max(max_fun, feval)
        total += feval
    return max_fun if rho is None else max_fun + rho * total


def jmetal_weighted_sum(f, w, z, nad, normalize=False, eps=1e-6):  # type: ignore[no-untyped-def]
    return sum(w[n] * (_value(f, z, nad, n, True, eps) if normalize else f[n]) for n in range(len(f)))


def jmetal_pbi(f, w, z, nad, theta, normalize=False, eps=1e-6, inverted=False):  # type: ignore[no-untyped-def]
    def value(n: int) -> float:
        if inverted:
            return float((nad[n] - f[n]) / (nad[n] - z[n] + eps)) if normalize else float(nad[n] - f[n])
        return _value(f, z, nad, n, normalize, eps)

    nl = float(np.sqrt(sum(w[n] ** 2 for n in range(len(f)))))
    d1 = abs(sum(value(n) * w[n] for n in range(len(f)))) / nl
    d2 = float(np.sqrt(sum((value(n) - d1 * (w[n] / nl)) ** 2 for n in range(len(f)))))
    return theta * d2 - d1 if inverted else d1 + theta * d2


def _reference(name: str, params: dict[str, Any], f, w, z, nad):  # type: ignore[no-untyped-def]
    normalize, eps = bool(params.get("normalize", False)), float(params.get("epsilon", 1e-6))
    if name == "tchebycheff":
        return jmetal_tschebyscheff(f, w, z, nad, normalize, eps)
    if name == "augmented_tchebycheff":
        return jmetal_tschebyscheff(f, w, z, nad, normalize, eps, rho=params.get("rho", 0.0001))
    if name == "modified_tchebycheff":  # VAMOS's: the augmented formula with rho = 0.001.
        return jmetal_tschebyscheff(f, w, z, nad, normalize, eps, rho=params.get("rho", 0.001))
    if name == "weighted_sum":
        return jmetal_weighted_sum(f, w, z, nad, normalize, eps)
    if name == "pbi":
        return jmetal_pbi(f, w, z, nad, params.get("theta", 5.0), normalize, eps)
    return jmetal_pbi(f, w, z, nad, params.get("theta", 0.1), normalize, eps, inverted=True)


AGGREGATIONS: list[tuple[str, dict[str, Any]]] = [
    ("augmented_tchebycheff", {}),
    ("augmented_tchebycheff", {"rho": 0.05, "normalize": True, "epsilon": 1e-3}),
    ("inverted_pbi", {}),
    ("inverted_pbi", {"theta": 2.0, "normalize": True}),
    ("tchebycheff", {"normalize": True}),
    ("weighted_sum", {"normalize": True, "epsilon": 0.5}),
    ("pbi", {"theta": 5.0, "normalize": True}),
    ("modified_tchebycheff", {"rho": 0.01, "normalize": True}),
]
IDS = [f"{name}-{'-'.join(f'{k}={v}' for k, v in params.items())}" for name, params in AGGREGATIONS]


# ---- Hand-computed values --------------------------------------------------------------------


def test_augmented_tchebycheff_by_hand() -> None:
    # |f - z| = [0.5, 1.0]; weighted [0.125, 0.75]; max 0.75 plus 1e-4 * 0.875 (jMetal's rho).
    value = augmented_tchebycheff(np.array([[0.5, 2.0]]), np.array([[0.25, 0.75]]), np.array([0.0, 1.0]))
    assert value[0] == pytest.approx(0.75 + 0.0001 * 0.875, abs=1e-15)
    # A zero weight counts its difference times 1e-4, as jMetal does.
    value = augmented_tchebycheff(np.array([[0.5, 2.0]]), np.array([[0.0, 1.0]]), np.array([0.0, 1.0]), rho=0.5)
    assert value[0] == pytest.approx(1.0 + 0.5 * (0.5e-4 + 1.0), abs=1e-15)


def test_inverted_pbi_by_hand() -> None:
    # u = z^nad - f = [1, 2]; w = [0.6, 0.8] (unit); d1 = 2.2; d2 = |[1, 2] - 2.2 w| = |[-0.32, 0.24]| = 0.4.
    value = inverted_pbi(np.array([[1.0, 1.0]]), np.array([[0.6, 0.8]]), np.array([2.0, 3.0]), theta=0.1)
    assert value[0] == pytest.approx(0.1 * 0.4 - 2.2, abs=1e-12)
    assert inverted_pbi(np.array([[1.0, 1.0]]), np.array([[0.6, 0.8]]), np.array([2.0, 3.0]))[0] == pytest.approx(0.04 - 2.2)


def test_normalized_tchebycheff_by_hand() -> None:
    # z* = [1, 0], z^nad = [3, 4]: f = [2, 3] becomes [1 / (2 + eps), 3 / (4 + eps)].
    aggregate = build_aggregator("tchebycheff", {"normalize": True, "epsilon": 0.5})
    value = aggregate(np.array([[2.0, 3.0]]), np.array([[0.5, 0.5]]), np.array([1.0, 0.0]), np.array([3.0, 4.0]))
    assert value[0] == pytest.approx(max(0.5 / 2.5, 0.5 * 3.0 / 4.5))


@pytest.mark.parametrize(("name", "params"), AGGREGATIONS, ids=IDS)
def test_the_aggregations_follow_jmetal(name: str, params: dict[str, Any]) -> None:
    rng = np.random.default_rng(3)
    for n_obj in (2, 3, 5):
        F = rng.uniform(0.0, 4.0, size=(25, n_obj))
        W = rng.dirichlet(np.ones(n_obj), size=25)
        W[0, 0] = 0.0
        ideal = F.min(axis=0) - 0.1
        nadir = F.max(axis=0) + 0.2
        values = build_aggregator(name, params)(F, W, ideal, nadir)
        expected = [_reference(name, params, F[i], W[i], ideal, nadir) for i in range(25)]
        np.testing.assert_allclose(values, expected, rtol=1e-12, atol=1e-12)


def test_the_new_names_and_their_defaults() -> None:
    assert {"augmented_tchebycheff", "inverted_pbi"} <= set(aggregation_methods())
    assert resolve_aggregation_spec("augmented_tchebycheff", {}) == (AGG_AUGMENTED_TCHEBYCHEFF, 5.0, 0.0001)
    assert resolve_aggregation_spec("inverted_pbi", {}) == (AGG_INVERTED_PBI, 0.1, 0.001)
    assert resolve_aggregation_spec("inverted_pbi", {"theta": 3}) == (AGG_INVERTED_PBI, 3.0, 0.001)


# ---- The neighbourhood update (compiled and Python) ----------------------------------------


def _state(name: str, params: dict[str, Any], rng: np.random.Generator, n_obj: int) -> MOEADState:
    pop = 12
    weights = rng.dirichlet(np.ones(n_obj), size=pop)
    weights[0, 1] = 0.0
    weights[0] /= weights[0].sum()
    agg_id, theta, rho = resolve_aggregation_spec(name, params)
    F = rng.uniform(0.0, 3.0, size=(pop, n_obj))
    state = MOEADState(
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
        aggregator=build_aggregator(name, params),
        aggregation_id=agg_id,
        aggregation_theta=theta,
        aggregation_rho=rho,
        aggregation_normalize=bool(params.get("normalize", False)),
        aggregation_epsilon=float(params.get("epsilon", 1e-6)),
        nadir=F.max(axis=0) + 0.3,
        replace_limit=4,
        batch_size=1,
    )
    return state


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "python"])
@pytest.mark.parametrize(("name", "params"), AGGREGATIONS, ids=IDS)
def test_the_neighbourhood_update_replaces_by_the_reference_values(
    name: str, params: dict[str, Any], compiled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vamos.engine.algorithm.moead import neighborhood

    if compiled:
        pytest.importorskip("numba")
        assert neighborhood.get_update_neighborhood_numba() is not None
    else:
        monkeypatch.setattr(neighborhood, "get_update_neighborhood_numba", lambda: None)
    rng = np.random.default_rng(11)
    for n_obj in (2, 3):
        for _ in range(20):
            state = _state(name, params, rng, n_obj)
            child_f = rng.uniform(0.0, 3.0, size=n_obj)
            order = rng.permutation(state.pop_size)
            expected_F = state.F.copy()
            replaced = 0
            for k in order:
                child = _reference(name, params, child_f, state.weights[k], state.ideal, state.nadir)
                current = _reference(name, params, expected_F[k], state.weights[k], state.ideal, state.nadir)
                if child < current:
                    expected_F[k] = child_f
                    replaced += 1
                    if replaced >= state.replace_limit:
                        break
            update_neighborhood(
                st=state, idx=0, child=np.array([99.0]), child_f=child_f, child_g=None, cv_penalty=0.0, candidate_order=order
            )
            np.testing.assert_array_equal(state.F, expected_F)


# ---- The nadir point -----------------------------------------------------------------------------


def test_the_nadir_point_is_the_nadir_of_the_non_dominated_solutions() -> None:
    front, cv = nondominated_front(np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0], [3.0, 3.0], [5.0, 5.0], [2.0, 2.0]]), None)
    assert sorted(map(tuple, front)) == [(1.0, 4.0), (2.0, 2.0), (4.0, 1.0)]
    np.testing.assert_array_equal(nadir_point(front), [4.0, 4.0])
    front, cv = add_to_front(front, cv, np.array([1.5, 1.5]), 0.0)
    assert sorted(map(tuple, front)) == [(1.0, 4.0), (1.5, 1.5), (4.0, 1.0)]
    front, cv = add_to_front(front, cv, np.array([5.0, 5.0]), 0.0)
    np.testing.assert_array_equal(nadir_point(front), [4.0, 4.0])
    front, cv = add_to_front(front, cv, np.array([0.5, 0.5]), 0.0)
    np.testing.assert_array_equal(front, [[0.5, 0.5]])
    np.testing.assert_array_equal(nadir_point(front), [0.5, 0.5])


def test_the_nadir_point_compares_violations_first() -> None:
    # jMetal's DominanceWithConstraintsComparator: the less violated dominates; equal violations compare objectives.
    F = np.array([[1.0, 1.0], [3.0, 3.0], [4.0, 0.5], [0.0, 0.0]])
    front, cv = nondominated_front(F, np.array([2.0, 0.0, 0.0, 1.0]))
    assert sorted(map(tuple, front)) == [(3.0, 3.0), (4.0, 0.5)]
    np.testing.assert_array_equal(nadir_point(front), [4.0, 3.0])
    front, cv = nondominated_front(np.array([[1.0, 1.0], [0.0, 0.0], [5.0, 5.0]]), np.array([2.0, 1.0, 1.0]))
    assert sorted(map(tuple, front)) == [(0.0, 0.0)]
    front, cv = add_to_front(front, cv, np.array([9.0, 9.0]), 0.0)
    np.testing.assert_array_equal(front, [[9.0, 9.0]])


def test_the_run_tracks_the_nadir_point_of_every_evaluated_solution() -> None:
    cfg = _config("inverted_pbi", {"theta": 0.5})
    problem = ZDT1Problem(n_var=8)
    algorithm = MOEAD(cfg.to_dict(), kernel=NumPyKernel())
    algorithm.initialize(problem, ("max_evaluations", 400), seed=3)
    state = algorithm._st
    assert state is not None and state.nadir is not None
    seen = [state.F.copy()]
    front, _ = nondominated_front(state.F, None)
    np.testing.assert_array_equal(state.nadir, nadir_point(front))
    from vamos.foundation.eval import EvaluationResult

    for _ in range(30):
        X = algorithm.ask()
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)
        algorithm.tell(EvaluationResult(F=out["F"], G=None))
        seen.append(out["F"])
    everything = np.vstack(seen)
    front, _ = nondominated_front(everything, None)
    np.testing.assert_array_equal(state.nadir, nadir_point(front))
    assert state.nadir_front is not None
    assert sorted(map(tuple, state.nadir_front)) == sorted(map(tuple, front))


# ---- Runs ---------------------------------------------------------------------------------------


def _config(name: str, params: dict[str, Any], *, batch_size: int = 1, constrained: bool = False) -> MOEADConfig:
    builder = (
        MOEADConfig.builder()
        .pop_size(20)
        .batch_size(batch_size)
        .neighbor_size(6)
        .delta(0.9)
        .replace_limit(2)
        .crossover("de", cr=1.0, f=0.5)
        .mutation("pm", prob="1/n", eta=20.0)
        .aggregation(name, **params)
        .weight_vectors(divisions=19)
    )
    if constrained:
        builder.subproblem_update_criterion("improved_epsilon")
    return builder.build()


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize(("name", "params"), AGGREGATIONS, ids=IDS)
def test_the_engines_give_the_same_run(name: str, params: dict[str, Any], batch_size: int) -> None:
    pytest.importorskip("numba")
    from vamos.foundation.kernel.numba_backend import NumbaKernel

    cfg = _config(name, params, batch_size=batch_size).to_dict()
    problem = ZDT1Problem(n_var=8)
    expected = MOEAD(dict(cfg), kernel=NumPyKernel()).run(problem, ("max_evaluations", 603), seed=5)
    algorithm = MOEAD(dict(cfg), kernel=NumbaKernel())
    actual = algorithm.run(problem, ("max_evaluations", 603), seed=5)
    for key in ("X", "F"):
        assert actual["population"][key].tobytes() == expected["population"][key].tobytes()
    assert actual["checkpoint"]["rng_state"] == expected["checkpoint"]["rng_state"]
    # The compiled one-offspring step needs no nadir point, so it runs the augmented Tchebycheff aggregation only.
    needs_nadir = name == "inverted_pbi" or params.get("normalize", False)
    if batch_size == 1 and NumbaKernel().variation_kernels() is not None:
        assert (algorithm._steady is None) == bool(needs_nadir)


def test_augmented_tchebycheff_is_the_formula_of_modified_tchebycheff_with_jmetals_rho() -> None:
    problem = ZDT1Problem(n_var=8)

    def run(name: str, params: dict[str, Any]) -> bytes:
        result = MOEAD(_config(name, params).to_dict(), kernel=NumPyKernel()).run(problem, ("max_evaluations", 603), seed=5)
        return bytes(result["population"]["F"].tobytes())

    assert run("augmented_tchebycheff", {}) == run("modified_tchebycheff", {"rho": 0.0001})
    assert run("augmented_tchebycheff", {"rho": 0.001}) == run("modified_tchebycheff", {})


CHANGING = [case for case in AGGREGATIONS if case[0] == "inverted_pbi" or case[1].get("normalize")]


@pytest.mark.parametrize(("name", "params"), CHANGING, ids=[IDS[AGGREGATIONS.index(case)] for case in CHANGING])
def test_each_option_changes_the_search(name: str, params: dict[str, Any]) -> None:
    problem = ZDT1Problem(n_var=8)
    run = MOEAD(_config(name, params).to_dict(), kernel=NumPyKernel()).run(problem, ("max_evaluations", 603), seed=5)
    plain_name = {"inverted_pbi": "pbi"}.get(name, name)
    plain_params = {key: value for key, value in params.items() if key not in {"normalize", "epsilon"}}
    plain = MOEAD(_config(plain_name, plain_params).to_dict(), kernel=NumPyKernel()).run(problem, ("max_evaluations", 603), seed=5)
    assert run["evaluations"] == 603
    assert run["population"]["F"].tobytes() != plain["population"]["F"].tobytes()


class _StopAtGeneration:
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
def test_a_resumed_run_keeps_the_nadir_point(batch_size: int) -> None:
    cfg = _config("tchebycheff", {"normalize": True}, batch_size=batch_size).to_dict()
    problem = ZDT1Problem(n_var=8)

    def run(**kwargs: Any) -> dict[str, Any]:
        return MOEAD(dict(cfg), kernel=NumPyKernel()).run(problem, ("max_evaluations", 800), seed=8, **kwargs)

    whole = run()
    stopped = run(live_viz=_StopAtGeneration(10))
    checkpoint = stopped["checkpoint"]
    assert 20 < checkpoint["n_eval"] < 800
    assert checkpoint["extra"]["nadir_front"].shape[1] == 2
    resumed = run(checkpoint=checkpoint)
    for key in ("X", "F"):
        assert resumed["population"][key].tobytes() == whole["population"][key].tobytes()
    np.testing.assert_array_equal(resumed["checkpoint"]["extra"]["nadir_front"], whole["checkpoint"]["extra"]["nadir_front"])


def test_the_subproblem_update_criteria_aggregate_with_the_nadir_point() -> None:
    from vamos.problems import LIRCMOP

    problem = LIRCMOP(2, n_var=10)
    base = dataclasses.replace(MOEADConfig.default(pop_size=30, n_var=10, n_obj=2), subproblem_update_criterion="violation_threshold")
    runs = {}
    for label, aggregation in {
        "plain": ("tchebycheff", {}),
        "normalized": ("tchebycheff", {"normalize": True}),
        "ipbi": ("inverted_pbi", {}),
    }.items():
        cfg = dataclasses.replace(base, aggregation=aggregation)
        runs[label] = MOEAD(cfg.to_dict(), kernel=NumPyKernel()).run(problem, ("max_evaluations", 900), seed=4)
        assert runs[label]["evaluations"] == 900
    assert runs["normalized"]["population"]["F"].tobytes() != runs["plain"]["population"]["F"].tobytes()
    assert runs["ipbi"]["population"]["F"].tobytes() != runs["plain"]["population"]["F"].tobytes()


def test_the_options_round_trip_and_replay_exactly(tmp_path: Any) -> None:
    cfg = _config("inverted_pbi", {"theta": 1.5, "normalize": True, "epsilon": 0.01})
    assert cfg.aggregation == ("inverted_pbi", {"theta": 1.5, "normalize": True, "epsilon": 0.01})
    assert MOEADConfig.from_dict(cfg.to_dict()) == cfg
    result = vamos.optimize("zdt1", n_var=8, algorithm="moead", algorithm_config=cfg, max_evaluations=120, seed=2, engine="numpy")
    stored = vamos.save_result(result, tmp_path / "source")
    resolved = vamos.load_run(stored.root).manifest.resolved_spec
    name, params = resolved["algorithm"]["config"]["aggregation"]
    assert (name, dict(params)) == ("inverted_pbi", {"epsilon": 0.01, "normalize": True, "theta": 1.5})
    assert vamos.verify_run(stored.root).effective_replayability == "exact"
    assert vamos.reproduce(stored.root, output=tmp_path / "replay").exact is True


# ---- Validation -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "params", "message"),
    [
        ("tchebycheff", {"normalize": "yes"}, "aggregation parameter normalize must be True or False"),
        ("pbi", {"normalize": 1}, "aggregation parameter normalize must be True or False"),
        ("pbi", {"normalize": True, "epsilon": 0.0}, "aggregation parameter epsilon must be finite and > 0"),
        ("pbi", {"normalize": True, "epsilon": float("nan")}, "aggregation parameter epsilon must be finite and > 0"),
        ("weighted_sum", {"epsilon": 1e-3}, "epsilon applies only with normalize=True"),
        ("inverted_pbi", {"rho": 0.1}, "aggregation 'inverted_pbi' takes theta, normalize, epsilon; got 'rho'"),
        ("augmented_tchebycheff", {"theta": 5.0}, "aggregation 'augmented_tchebycheff' takes rho, normalize, epsilon; got 'theta'"),
        ("inverted_pbi", {"theta": -1.0}, "aggregation 'inverted_pbi' parameter theta must be finite and >= 0"),
        ("augmented_tchebycheff", {"rho": float("inf")}, "aggregation 'augmented_tchebycheff' parameter rho must be finite and >= 0"),
    ],
)
def test_invalid_aggregation_parameters_are_rejected(name: str, params: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        _config(name, params)
    cfg = _config("tchebycheff", {}).to_dict()
    cfg["aggregation"] = (name, params)
    with pytest.raises(ConfigurationError, match=message):
        MOEAD(cfg, kernel=NumPyKernel()).run(ZDT1Problem(n_var=8), ("max_evaluations", 60), seed=1)
