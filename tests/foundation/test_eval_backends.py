import numpy as np
import pytest

from vamos.foundation.eval.backends import (
    DaskEvalBackend,
    MultiprocessingEvalBackend,
    SerialEvalBackend,
    resolve_eval_strategy,
)
from vamos.foundation.eval.population import evaluate_population_with_constraints
from vamos.foundation.exceptions import EvaluationError


class DummyProblem:
    def __init__(self):
        self.n_var = 2
        self.n_obj = 1
        self.n_constraints = 0
        self.xl = -5.0
        self.xu = 5.0
        self.encoding = "real"

    def evaluate(self, X, out):
        out["F"] = np.sum(X * X, axis=1, keepdims=True)


def test_serial_eval_strategy_matches_direct():
    prob = DummyProblem()
    X = np.array([[1.0, 2.0], [0.5, -0.5]])
    backend = SerialEvalBackend()

    res = backend.evaluate(X, prob)

    expected = np.sum(X * X, axis=1, keepdims=True)
    np.testing.assert_allclose(res.F, expected)
    assert res.G is None


def test_multiprocessing_eval_strategy_matches_serial():
    prob = DummyProblem()
    X = np.array([[1.0, 2.0], [0.5, -0.5], [3.0, 0.0]])
    serial = SerialEvalBackend().evaluate(X, prob)
    mp = MultiprocessingEvalBackend(n_workers=2).evaluate(X, prob)

    np.testing.assert_allclose(mp.F, serial.F)
    assert mp.G is None


class MissingWriteProblem(DummyProblem):
    def evaluate(self, X, out):
        return None


class WrongShapeProblem(DummyProblem):
    def evaluate(self, X, out):
        out["F"] = np.zeros((X.shape[0], 2))


class NonFiniteProblem(DummyProblem):
    def evaluate(self, X, out):
        out["F"][:, 0] = np.nan


def test_population_evaluation_rejects_missing_objective_write():
    with pytest.raises(EvaluationError, match="non-finite"):
        evaluate_population_with_constraints(MissingWriteProblem(), np.zeros((2, 2)))


def test_population_evaluation_rejects_wrong_objective_shape():
    with pytest.raises(EvaluationError, match="shape"):
        evaluate_population_with_constraints(WrongShapeProblem(), np.zeros((2, 2)))


def test_population_evaluation_rejects_non_finite_objectives():
    with pytest.raises(EvaluationError, match="non-finite"):
        evaluate_population_with_constraints(NonFiniteProblem(), np.zeros((2, 2)))


def test_dask_backend_requires_explicit_fallback_when_not_connected(monkeypatch):
    distributed = pytest.importorskip("dask.distributed")

    def _no_client():
        raise ValueError("no active client")

    monkeypatch.setattr(distributed, "get_client", _no_client)
    backend = DaskEvalBackend(client=None)

    with pytest.raises(EvaluationError, match="fallback_to_serial=True"):
        backend.evaluate(np.zeros((2, 2)), DummyProblem())


def test_dask_backend_serial_fallback_is_opt_in(monkeypatch):
    distributed = pytest.importorskip("dask.distributed")

    def _no_client():
        raise ValueError("no active client")

    monkeypatch.setattr(distributed, "get_client", _no_client)
    backend = DaskEvalBackend(client=None, fallback_to_serial=True)

    result = backend.evaluate(np.array([[1.0, 2.0], [0.5, -0.5]]), DummyProblem())

    np.testing.assert_allclose(result.F, np.array([[5.0], [0.5]]))
    assert result.G is None


def test_dask_strategy_discovers_active_client():
    distributed = pytest.importorskip("dask.distributed")
    Client = distributed.Client
    LocalCluster = distributed.LocalCluster

    X = np.array([[1.0, 2.0], [0.5, -0.5]])
    with LocalCluster(n_workers=1, threads_per_worker=1, processes=False, dashboard_address=None) as cluster:
        with Client(cluster) as client:
            backend = resolve_eval_strategy("dask")
            assert isinstance(backend, DaskEvalBackend)
            assert backend.client is client

            result = backend.evaluate(X, DummyProblem())

    np.testing.assert_allclose(result.F, np.array([[5.0], [0.5]]))
    assert result.G is None
