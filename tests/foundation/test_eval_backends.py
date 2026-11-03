import numpy as np
import pytest

from vamos.foundation.eval.backends import (
    DaskEvalBackend,
    MultiprocessingEvalBackend,
    SerialEvalBackend,
    resolve_eval_strategy,
)
from vamos.foundation.eval.population import evaluate_population_with_constraints
from vamos.foundation.exceptions import ConfigurationError, EvaluationError


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


def test_dask_backend_errors_suggest_a_scheduler(monkeypatch):
    distributed = pytest.importorskip("dask.distributed")

    def _no_client():
        raise ValueError("no active client")

    monkeypatch.setattr(distributed, "get_client", _no_client)
    backend = DaskEvalBackend(client=None)

    with pytest.raises(EvaluationError) as excinfo:
        backend.evaluate(np.zeros((2, 2)), DummyProblem())
    assert "evaluate()" not in (excinfo.value.suggestion or "")
    assert "Dask" in (excinfo.value.suggestion or "")


def test_dask_backend_reports_an_unreachable_scheduler(monkeypatch):
    distributed = pytest.importorskip("dask.distributed")

    def _unreachable(address):
        raise OSError(f"Timed out trying to connect to {address}")

    monkeypatch.setattr(distributed, "Client", _unreachable)

    with pytest.raises(ConfigurationError, match="tcp://127.0.0.1:1") as excinfo:
        DaskEvalBackend(address="tcp://127.0.0.1:1")
    assert "Dask" in (excinfo.value.suggestion or "")


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


class SlowProblem(DummyProblem):
    def evaluate(self, X, out):
        import time

        time.sleep(2.0)
        out["F"] = np.sum(X * X, axis=1, keepdims=True)


def _spy_on_pools(monkeypatch):
    """Replace the backend's ProcessPoolExecutor with one that records its instances."""
    from concurrent.futures import ProcessPoolExecutor

    import vamos.foundation.eval.backends as backends

    created = []

    class SpyExecutor(ProcessPoolExecutor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.shut_down = False
            created.append(self)

        def shutdown(self, *args, **kwargs):
            self.shut_down = True
            return super().shutdown(*args, **kwargs)

    monkeypatch.setattr(backends, "ProcessPoolExecutor", SpyExecutor)
    return created


def test_multiprocessing_backend_reuses_one_pool_across_calls(monkeypatch):
    created = _spy_on_pools(monkeypatch)
    prob = DummyProblem()
    X = np.array([[1.0, 2.0], [0.5, -0.5], [3.0, 0.0], [0.0, 1.0]])
    backend = MultiprocessingEvalBackend(n_workers=2)

    for _ in range(3):
        result = backend.evaluate(X, prob)
        np.testing.assert_allclose(result.F, np.sum(X * X, axis=1, keepdims=True))

    # VAMOS 1.1.0 started and stopped a pool of worker processes in every call, that is, every generation.
    assert len(created) == 1
    assert not created[0].shut_down
    backend.close()
    assert created[0].shut_down


def test_multiprocessing_backend_is_a_context_manager(monkeypatch):
    created = _spy_on_pools(monkeypatch)
    X = np.array([[1.0, 2.0], [0.5, -0.5]])

    with MultiprocessingEvalBackend(n_workers=2) as backend:
        backend.evaluate(X, DummyProblem())

    assert len(created) == 1 and created[0].shut_down


def test_multiprocessing_backend_timeout_raises():
    import time

    backend = MultiprocessingEvalBackend(n_workers=2, timeout=0.2)
    X = np.array([[1.0, 2.0], [0.5, -0.5]])

    start = time.perf_counter()
    with pytest.raises(EvaluationError, match="timeout"):
        backend.evaluate(X, SlowProblem())
    elapsed = time.perf_counter() - start
    backend.close()

    # VAMOS 1.1.0 applied the timeout only after every chunk had finished, so it never fired.
    assert elapsed < 1.5


@pytest.mark.parametrize("n_workers", [0, -1, 1.5, True])
def test_multiprocessing_backend_rejects_non_positive_workers(n_workers):
    # VAMOS 1.1.0 read n_workers=0 as every core and a negative count as one worker.
    with pytest.raises(ConfigurationError, match="n_workers"):
        MultiprocessingEvalBackend(n_workers=n_workers)


@pytest.mark.parametrize("timeout", [0, -1.0])
def test_multiprocessing_backend_rejects_non_positive_timeout(timeout):
    with pytest.raises(ConfigurationError, match="timeout"):
        MultiprocessingEvalBackend(n_workers=2, timeout=timeout)


def test_multiprocessing_backend_bounds_its_default_workers(monkeypatch):
    import vamos.foundation.eval.backends as backends

    monkeypatch.setattr(backends.os, "cpu_count", lambda: 64)

    assert MultiprocessingEvalBackend().n_workers == 8


class _InlineFuture:
    def __init__(self, fn, args):
        try:
            self._value, self._error = fn(*args), None
        except Exception as exc:  # noqa: BLE001 - re-raised by gather, like a Dask future
            self._value, self._error = None, exc


class _FakeDaskClient:
    """Runs submitted tasks inline and re-raises their exceptions from gather, as dask.distributed does."""

    def scheduler_info(self):
        return {"workers": {"tcp://worker": {}}}

    def submit(self, fn, *args):
        return _InlineFuture(fn, args)

    def gather(self, futures):
        for future in futures:
            if future._error is not None:
                raise future._error
        return [future._value for future in futures]


class _CountingNonFiniteProblem(NonFiniteProblem):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def evaluate(self, X, out):
        self.calls += 1
        super().evaluate(X, out)


@pytest.mark.parametrize("fallback_to_serial", [False, True])
def test_dask_backend_propagates_problem_evaluation_error(fallback_to_serial):
    problem = _CountingNonFiniteProblem()
    backend = DaskEvalBackend(client=_FakeDaskClient(), fallback_to_serial=fallback_to_serial)

    with pytest.raises(EvaluationError, match="non-finite") as excinfo:
        backend.evaluate(np.zeros((2, 2)), problem)

    # The problem's own error is not wrapped as a scheduler failure, and it is not evaluated again in serial.
    assert "DaskEvalBackend evaluation failed" not in str(excinfo.value)
    assert problem.calls == 1


class _ClosingSpyBackend(SerialEvalBackend):
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


def test_optimize_closes_only_the_backends_it_builds(monkeypatch):
    import vamos
    import vamos.experiment.optimize as optimize_module

    built = []

    def _resolve(name, **kwargs):
        built.append(_ClosingSpyBackend())
        return built[-1]

    monkeypatch.setattr(optimize_module, "resolve_eval_strategy", _resolve)

    vamos.optimize("zdt1", algorithm="nsgaii", pop_size=10, max_evaluations=20, eval_strategy="multiprocessing", seed=1)
    # A pool built from a name lives as long as the run, so optimize closes it.
    assert len(built) == 1 and built[0].closed == 1

    own = _ClosingSpyBackend()
    vamos.optimize("zdt1", algorithm="nsgaii", pop_size=10, max_evaluations=20, eval_strategy=own, seed=1)
    # A backend the caller passes stays open for the caller to reuse.
    assert own.closed == 0
