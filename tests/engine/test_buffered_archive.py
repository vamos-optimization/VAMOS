"""The buffered UnboundedArchive keeps offered rows and merges them when read, with the rows of one update per call."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

from vamos.engine.algorithm.components import archive_buffer
from vamos.engine.algorithm.components.archive import UnboundedArchive
from vamos.engine.algorithm.components.archive_core import _BaseArchive
from vamos.engine.algorithm.components.state import AlgorithmState, ArchiveRows, record_archive_rows
from vamos.engine.algorithm.nsgaii.state import NSGAIIState
from vamos.experiment.unified import optimize


def _stream(rng: np.random.Generator, rows: int, n_obj: int, *, constrained: bool, nan: bool = False) -> tuple[np.ndarray, ...]:
    """Rows as an evolving population offers them: improving objectives, ties, duplicates and, optionally, NaN and infeasible rows."""
    drift = np.linspace(1.0, 0.0, rows)[:, None]
    F = np.round(rng.random((rows, n_obj)) * 8.0) / 8.0 + drift  # coarse values: ties and duplicates
    X = rng.random((rows, 3))
    G = None
    if constrained:
        G = rng.normal(0.0, 1.0, size=(rows, 2))
        G[: rows // 5] = np.abs(G[: rows // 5]) + 0.1  # an infeasible start, as on a constrained problem
    if nan:
        F[rng.choice(rows, size=rows // 50, replace=False), 0] = np.nan
    return X, F, G


def _offer(
    archive: UnboundedArchive, X: np.ndarray, F: np.ndarray, G: np.ndarray | None, batches: list[int], reads: set[int]
) -> list[tuple[Any, ...]]:
    seen = []
    start = 0
    for k, size in enumerate(batches):
        rows = slice(start, start + size)
        archive.add(X[rows], F[rows], None if G is None else G[rows])
        start += size
        if k in reads:
            seen.append((*archive.contents(), archive.constraint_values()))
    seen.append((*archive.contents(), archive.constraint_values()))
    return seen


@pytest.mark.parametrize("constrained", [False, True])
@pytest.mark.parametrize("nan", [False, True])
@pytest.mark.parametrize("n_obj", [2, 3, 5])
def test_buffered_contents_equal_one_update_per_call(constrained: bool, nan: bool, n_obj: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(archive_buffer, "PENDING_ROWS", 64)  # several merges along the stream
    rng = np.random.default_rng(n_obj + 10 * constrained + 100 * nan)
    X, F, G = _stream(rng, 3000, n_obj, constrained=constrained, nan=nan)
    batches = [int(size) for size in rng.integers(1, 120, size=60)]
    batches.append(len(F) - sum(batches))
    reads = {5, 17, 40}
    expected = _offer(UnboundedArchive(3, n_obj, float, n_con=2 if constrained else None, buffered=False), X, F, G, batches, reads)
    actual = _offer(UnboundedArchive(3, n_obj, float, n_con=2 if constrained else None), X, F, G, batches, reads)
    for got, want in zip(actual, expected, strict=True):
        for got_part, want_part in zip(got, want, strict=True):
            if want_part is None:
                assert got_part is None
            else:
                np.testing.assert_array_equal(got_part, want_part)


def test_rows_are_kept_only_while_the_contents_are_clean() -> None:
    archive = UnboundedArchive(1, 2, float, n_con=1)
    archive.add(np.zeros((2, 1)), np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([[1.0], [2.0]]))  # infeasible only
    assert not archive._pending  # no feasible row yet: merged at once
    archive.add(np.zeros((1, 1)), np.array([[0.5, 0.5]]), np.array([[-1.0]]))
    archive.add(np.zeros((1, 1)), np.array([[0.4, 0.4]]), np.array([[-1.0]]))
    assert archive._pending_rows == 1  # the archive is feasible and clean: kept
    X, F = archive.contents()
    np.testing.assert_array_equal(F, [[0.4, 0.4]])
    assert not archive._pending


def test_kept_rows_are_copies() -> None:
    archive = UnboundedArchive(1, 2, float)
    archive.update(np.zeros((1, 1)), np.array([[1.0, 1.0]]))
    F = np.array([[0.5, 0.5]])
    archive.add(np.zeros((1, 1)), F)
    F[0] = 9.0  # a population written after it was offered
    np.testing.assert_array_equal(archive.contents()[1], [[0.5, 0.5]])


def test_rows_mismatching_the_constraints_raise_when_offered() -> None:
    archive = UnboundedArchive(1, 2, float, n_con=1)
    archive.update(np.zeros((1, 1)), np.array([[1.0, 1.0]]), np.array([[-1.0]]))
    with pytest.raises(ValueError, match="constraint"):
        archive.add(np.zeros((1, 1)), np.array([[0.5, 0.5]]))


# ---- The state's archive fields ---------------------------------------------


def _all_states(cls: type) -> list[type]:
    return [cls, *[state for sub in cls.__subclasses__() for state in _all_states(sub)]]


def test_every_state_with_archive_fields_reads_them_lazily() -> None:
    import vamos.engine.algorithm  # noqa: F401  (imports every algorithm's state)

    for state_cls in {*_all_states(AlgorithmState), NSGAIIState}:
        for name in ("archive_X", "archive_F"):
            assert isinstance(inspect.getattr_static(state_cls, name), ArchiveRows), (state_cls, name)


def test_record_archive_rows_merges_on_first_read() -> None:
    archive = UnboundedArchive(1, 2, float)
    archive.update(np.zeros((1, 1)), np.array([[1.0, 1.0]]))
    state = AlgorithmState(X=np.zeros((1, 1)), F=np.zeros((1, 2)), G=None, rng=np.random.default_rng(0), archive_manager=archive)
    record_archive_rows(state, np.ones((2, 1)), np.array([[0.5, 0.5], [2.0, 2.0]]), None)
    assert archive._pending_rows == 2
    np.testing.assert_array_equal(state.archive_F, [[0.5, 0.5]])
    np.testing.assert_array_equal(state.archive_X, [[1.0]])
    assert not archive._pending

    record_archive_rows(state, np.zeros((1, 1)), np.array([[0.1, 0.9]]), None)
    state.archive_X = np.full((1, 1), 7.0)  # an assignment while stale merges first: archive_F stays current
    np.testing.assert_array_equal(state.archive_F, [[0.5, 0.5], [0.1, 0.9]])


def test_a_bounded_archive_is_updated_at_once() -> None:
    from vamos.engine.algorithm.components.archive import CrowdingDistanceArchive

    archive = CrowdingDistanceArchive(5, 1, 2, float)
    state = AlgorithmState(X=np.zeros((1, 1)), F=np.zeros((1, 2)), G=None, rng=np.random.default_rng(0), archive_manager=archive)
    record_archive_rows(state, np.zeros((1, 1)), np.array([[0.5, 0.5]]), None)
    assert "_archive_rows_stale" not in state.__dict__ or not state.__dict__["_archive_rows_stale"]
    np.testing.assert_array_equal(state.__dict__["_archive_F_rows"], [[0.5, 0.5]])


# ---- Runs ----------------------------------------------------------------------


ALGORITHMS = ["nsgaii", "nsgaiii", "moead", "smsemoa", "spea2", "ibea", "agemoea", "rvea", "ccmo", "ctaea", "top", "cmoeams", "pps"]


@pytest.mark.parametrize("problem", ["zdt1", "c1dtlz1"])
@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_runs_with_an_unbounded_archive_match_eager_updates(algorithm: str, problem: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import dataclasses

    from vamos.engine.algorithm.config.defaults import build_default_algorithm_config
    from vamos.engine.archive import ExternalArchiveConfig
    from vamos.foundation.problem.registry import make_problem_selection

    instance = make_problem_selection(problem).instantiate()
    pop_size = None if algorithm in ("moead", "nsgaiii", "pps", "rvea") else 24
    config = build_default_algorithm_config(algorithm, pop_size=pop_size, n_var=instance.n_var, n_obj=instance.n_obj)
    config = dataclasses.replace(config, external_archive=ExternalArchiveConfig(capacity=None))

    def run() -> Any:
        return optimize(problem, algorithm=algorithm, algorithm_config=config, max_evaluations=1200, seed=3, engine="numpy")

    merged: list[int] = []
    flush = _BaseArchive._flush

    def counting_flush(archive: _BaseArchive) -> None:
        merged.append(archive._pending_rows)
        flush(archive)

    monkeypatch.setattr(_BaseArchive, "_flush", counting_flush)
    buffered = run()
    if problem == "zdt1":  # C1-DTLZ1 finds no feasible row in this budget: nothing is kept
        assert max(merged) > 0  # the run kept rows for a later merge
    monkeypatch.setattr(_BaseArchive, "_bufferable", lambda self, F, G: False)
    eager = run()
    np.testing.assert_array_equal(buffered.X, eager.X)
    np.testing.assert_array_equal(buffered.F, eager.F)
