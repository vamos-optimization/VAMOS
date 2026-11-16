"""MOEA/D offers its external archive only the rows that a step changes.

Each run is compared, bit for bit, with the update it replaces: the whole population offered to
the archive after every step, merged and sorted (no insertion).
"""

from __future__ import annotations

import contextlib
import copy
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from vamos.engine.algorithm.components.archive_core import _BaseArchive
from vamos.engine.algorithm.config import MOEADConfig
from vamos.engine.algorithm.moead import MOEAD, moead, steady_state
from vamos.engine.algorithm.moead import state as moead_state
from vamos.engine.algorithm.moead.state import MOEADState
from vamos.engine.archive.factory import update_archive
from vamos.foundation.eval import EvaluationResult
from vamos.foundation.eval.backends import SerialEvalBackend
from vamos.foundation.kernel.numpy_backend import NumPyKernel
from vamos.foundation.problem.constrained_many import CDTLZProblem
from vamos.foundation.problem.lircmop import LIRCMOPProblem
from vamos.foundation.problem.zdt1 import ZDT1Problem

ARCHIVES: dict[str, dict[str, Any]] = {
    "unbounded": {"capacity": None},
    "crowding": {"capacity": 30, "pruning": "crowding"},
    "crowding-truncate": {"capacity": 30, "pruning": "crowding", "truncate_size": 20},
    "hv": {"capacity": 30, "pruning": "hv"},
    "knn": {"capacity": 30, "pruning": "knn"},
    "maxmin": {"capacity": 30, "pruning": "maxmin"},
    "ref_dirs": {"capacity": 30, "pruning": "ref_dirs", "rng_seed": 2},
    "decision": {"capacity": None, "deduplicate_in": "decision"},
    "tolerance": {"capacity": None, "objective_tolerance": 0.01},
}

PROBLEMS: dict[str, Any] = {
    "zdt1": lambda: ZDT1Problem(n_var=10),
    "c1dtlz1": lambda: CDTLZProblem("c1dtlz1", n_var=7, n_obj=3),
    "lircmop2": lambda: LIRCMOPProblem(2, n_var=10),
}


def _kernel(engine: str) -> Any:
    if engine == "numba":
        from vamos.foundation.kernel.numba_backend import NumbaKernel

        return NumbaKernel()
    if engine == "moocore":
        from vamos.foundation.kernel.moocore_backend import MooCoreKernel

        return MooCoreKernel()
    return NumPyKernel()


def _config(problem: Any, archive: str, **changes: Any) -> dict[str, Any]:
    pop_size = 40 if problem.n_obj == 2 else 36
    cfg = MOEADConfig.default(pop_size=pop_size, n_var=problem.n_var, n_obj=problem.n_obj).to_dict()
    if problem.n_constraints:
        cfg["constraint_mode"] = "feasibility"
    cfg["external_archive"] = dict(ARCHIVES[archive])
    cfg.update(changes)
    return cfg


def _offer_whole_population(st: MOEADState, baseline: object, *, changed: bool = True) -> None:
    update_archive(st, st.X, st.F, st.G)


@contextlib.contextmanager
def _whole_population_updates() -> Iterator[None]:
    """The archive update this change replaces: the whole population after every step, merged and sorted."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(moead, "update_moead_archive", _offer_whole_population)
        patch.setattr(steady_state, "update_moead_archive", _offer_whole_population)
        patch.setattr(_BaseArchive, "_insertable", lambda self, n_rows, pop_G: False)
        yield


class _OfferedRows:
    """Count the rows offered to the archives, to check that the runs offer fewer than the population."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[int] = []
        self.active = True
        update = _BaseArchive.update

        def counting_update(archive: _BaseArchive, X: Any, F: Any, G: Any = None) -> Any:
            if self.active:
                self.calls.append(int(np.asarray(F).shape[0]))
            return update(archive, X, F, G)

        monkeypatch.setattr(_BaseArchive, "update", counting_update)


def _run(cfg: dict[str, Any], problem: Any, engine: str, seed: int, termination: Any, **kwargs: Any) -> dict[str, Any]:
    return MOEAD(dict(cfg), kernel=_kernel(engine)).run(problem, termination=termination, seed=seed, **kwargs)


def _assert_same(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    def same(left: Any, right: Any) -> bool:
        if left is None or right is None:
            return left is None and right is None
        left, right = np.asarray(left), np.asarray(right)
        return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()

    for key in ("X", "F", "G"):
        assert same(expected.get(key), actual.get(key)), key
        assert same(expected["population"].get(key), actual["population"].get(key)), key
        assert same(expected["archive"].get(key), actual["archive"].get(key)), key
    left, right = expected["checkpoint"], actual["checkpoint"]
    for key in ("X", "F", "G", "archive_X", "archive_F", "archive_G"):
        assert same(left[key], right[key]), key
    for key in ("subproblem_order", "ideal"):
        assert same(left["extra"][key], right["extra"][key]), key
    assert right["rng_state"] == left["rng_state"]
    assert (right["generation"], right["n_eval"], right["extra"]["subproblem_cursor"]) == (
        left["generation"],
        left["n_eval"],
        left["extra"]["subproblem_cursor"],
    )
    assert (actual["evaluations"], actual["hv_reached"]) == (expected["evaluations"], expected["hv_reached"])


def _check(
    cfg: dict[str, Any], problem: str, engine: str, *, seed: int = 3, evals: int = 1500, offered: _OfferedRows | None = None, **kwargs: Any
) -> None:
    termination = kwargs.pop("termination", ("max_evaluations", evals))
    # Each run gets its own copy: a resumed run works in the checkpoint's arrays.
    actual = _run(cfg, PROBLEMS[problem](), engine, seed, termination, **copy.deepcopy(kwargs))
    if offered is not None:
        offered.active = False
    with _whole_population_updates():
        expected = _run(cfg, PROBLEMS[problem](), engine, seed, termination, **copy.deepcopy(kwargs))
    _assert_same(expected, actual)


@pytest.mark.parametrize("archive", sorted(ARCHIVES))
@pytest.mark.parametrize(("problem", "engine"), [("zdt1", "numba"), ("c1dtlz1", "numpy"), ("lircmop2", "moocore")])
def test_archive_matches_whole_population_updates(archive: str, problem: str, engine: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if engine == "numba":
        pytest.importorskip("numba")
    if engine == "moocore" or (archive == "hv" and problem == "c1dtlz1"):
        pytest.importorskip("moocore")  # the engine, and hypervolume pruning with three objectives
    offered = _OfferedRows(monkeypatch)
    cfg = _config(PROBLEMS[problem](), archive)
    _check(cfg, problem, engine, offered=offered)
    whole, steps = offered.calls.count(cfg["pop_size"]), 1 + 1500 - cfg["pop_size"]
    # Steps that change nothing are skipped; without exact objective deduplication (or a feasible row
    # yet, on C1-DTLZ1) the archive never covers the population, which it then gets whole.
    assert len(offered.calls) < steps
    if archive in ("decision", "tolerance") or problem == "c1dtlz1":
        assert whole == len(offered.calls)
    else:
        assert whole < len(offered.calls) * (0.1 if problem == "zdt1" else 0.7)


@pytest.mark.parametrize("archive", ["unbounded", "crowding", "ref_dirs"])
@pytest.mark.parametrize("variant", ["sbx-tchebycheff", "batch", "genealogy", "improved-epsilon"])
def test_reference_step_variants_match_whole_population_updates(archive: str, variant: str) -> None:
    problem = "c1dtlz1" if variant == "improved-epsilon" else "zdt1"
    changes: dict[str, Any] = {
        "sbx-tchebycheff": {"crossover": ("sbx", {"prob": 0.9, "eta": 20.0}), "aggregation": ("tchebycheff", {})},
        "batch": {"batch_size": 6},
        "genealogy": {"track_genealogy": True},
        "improved-epsilon": {
            "subproblem_update_criterion": ("improved_epsilon", {"tao": 0.1, "feasibility_ratio_threshold": 0.5, "decay_fraction": 0.8})
        },
    }[variant]
    _check(_config(PROBLEMS[problem](), archive, **changes), problem, "numpy")


@pytest.mark.parametrize("archive", ["unbounded", "crowding-truncate", "hv"])
@pytest.mark.parametrize("engine", ["numpy", "numba"])
def test_hv_termination_matches_whole_population_updates(archive: str, engine: str) -> None:
    if engine == "numba":
        pytest.importorskip("numba")
    cfg = _config(PROBLEMS["zdt1"](), archive)
    hv = ("hv", {"target_value": 120.2, "reference_point": [11.0, 11.0], "max_evaluations": 3000})
    actual = _run(cfg, PROBLEMS["zdt1"](), engine, 5, hv)
    assert actual["hv_reached"] and actual["evaluations"] < 2500
    _check(cfg, "zdt1", engine, seed=5, termination=hv)


@pytest.mark.parametrize("archive", ["unbounded", "crowding", "knn"])
@pytest.mark.parametrize("problem", ["zdt1", "lircmop2"])
def test_resumed_runs_match_whole_population_updates(archive: str, problem: str) -> None:
    cfg = _config(PROBLEMS[problem](), archive)
    first = _run(cfg, PROBLEMS[problem](), "numpy", 8, ("max_evaluations", 900))
    with _whole_population_updates():
        expected_first = _run(cfg, PROBLEMS[problem](), "numpy", 8, ("max_evaluations", 900))
    _assert_same(expected_first, first)
    _check(cfg, problem, "numpy", seed=8, evals=1800, checkpoint=first["checkpoint"])


@pytest.mark.parametrize("archive", ["unbounded", "crowding-truncate", "ref_dirs"])
def test_ask_tell_archive_after_every_step_matches_whole_population_updates(archive: str) -> None:
    problem = PROBLEMS["lircmop2"]()
    cfg = _config(problem, archive)
    runs = []
    for reference in (False, True):
        algorithm = MOEAD(dict(cfg), kernel=NumPyKernel())
        trace = []
        with _whole_population_updates() if reference else contextlib.nullcontext():
            algorithm.initialize(problem, termination=("max_evaluations", 700), seed=4)
            backend = SerialEvalBackend()
            while not algorithm.should_terminate():
                algorithm.tell(backend.evaluate(algorithm.ask(), problem), problem)
                st = algorithm._st
                assert st is not None and st.archive_manager is not None
                X, F = st.archive_manager.contents()
                G = st.archive_manager.constraint_values()
                trace.append((X.tobytes(), F.tobytes(), None if G is None else G.tobytes(), st.hv_points().tobytes()))
                assert st.archive_X is not None and st.archive_X.tobytes() == X.tobytes()
            runs.append((trace, algorithm.result()))
    (actual_trace, actual), (expected_trace, expected) = runs
    assert actual_trace == expected_trace
    _assert_same(expected, actual)


class _NonFiniteResults:
    """Serial evaluation that gives some children an infinite or NaN objective."""

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, X: np.ndarray, problem: Any) -> EvaluationResult:
        result = SerialEvalBackend().evaluate(X, problem)
        self.calls += 1
        F = result.F.copy()
        if self.calls in (300, 301, 302):
            F[0, 0] = -np.inf if self.calls == 300 else np.nan if self.calls == 301 else np.inf
        return EvaluationResult(F=F, G=result.G)


@pytest.mark.parametrize("archive", ["unbounded", "crowding"])
def test_non_finite_children_match_whole_population_updates(archive: str, monkeypatch: pytest.MonkeyPatch) -> None:
    finite_checks: list[bool] = []
    finite_rows = moead_state._finite_rows

    def recording_finite_rows(state: MOEADState, rows: Any) -> bool:
        finite = finite_rows(state, rows)
        finite_checks.append(finite)
        return finite

    monkeypatch.setattr(moead_state, "_finite_rows", recording_finite_rows)
    # With the weighted sum, a child with an objective at -inf replaces its neighbours.
    cfg = _config(PROBLEMS["zdt1"](), archive, aggregation=("weighted_sum", {}))
    _check(cfg, "zdt1", "numpy", seed=4, evals=900, eval_strategy=_NonFiniteResults())
    assert False in finite_checks and True in finite_checks
