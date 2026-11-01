"""Training failures in the machine-learning benchmark problems."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from vamos.foundation.exceptions import EvaluationError  # noqa: E402
from vamos.foundation.problem.real_world.feature_selection import FeatureSelectionProblem  # noqa: E402
from vamos.foundation.problem.real_world.hyperparam import HyperparameterTuningProblem  # noqa: E402


class _FailingModel:
    """A classifier whose ``fit`` raises ``error``."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def fit(self, X: np.ndarray, y: np.ndarray) -> _FailingModel:
        raise self.error

    def score(self, X: np.ndarray, y: np.ndarray) -> float:  # pragma: no cover - never reached
        return 1.0


def _feature_selection_with(monkeypatch: pytest.MonkeyPatch, make_model) -> FeatureSelectionProblem:  # type: ignore[no-untyped-def]
    import sklearn.linear_model

    monkeypatch.setattr(sklearn.linear_model, "LogisticRegression", lambda **kwargs: make_model())
    return FeatureSelectionProblem()


def _evaluate(problem, X: np.ndarray) -> np.ndarray:  # type: ignore[no-untyped-def]
    out = {"F": np.zeros((X.shape[0], 2))}
    problem.evaluate(X, out)
    return out["F"]


def test_ml_problems_reraise_unexpected_training_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _feature_selection_with(monkeypatch, lambda: _FailingModel(TypeError("bug in the model")))
    with pytest.raises(TypeError, match="bug in the model"):
        _evaluate(problem, np.ones((2, problem.n_var)))

    tuning = HyperparameterTuningProblem()
    tuning._make_model = lambda params: _FailingModel(TypeError("bug in the model"))
    with pytest.raises(TypeError, match="bug in the model"):
        _evaluate(tuning, np.array([[0.5, 0.5, 0.0, 3.0]]))


def test_ml_problems_warn_about_expected_training_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    from sklearn.linear_model import LogisticRegression

    calls = {"n": 0}

    def flaky() -> object:
        calls["n"] += 1
        return _FailingModel(ValueError("singular data")) if calls["n"] == 1 else LogisticRegression(max_iter=200, solver="liblinear")

    problem = _feature_selection_with(monkeypatch, flaky)
    with pytest.warns(RuntimeWarning, match=r"1 of 2 .*singular data"):
        F = _evaluate(problem, np.ones((2, problem.n_var)))

    assert F[0, 0] == 1.0
    assert F[1, 0] < 1.0


def test_ml_problems_fail_when_every_model_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _feature_selection_with(monkeypatch, lambda: _FailingModel(ValueError("singular data")))
    with pytest.raises(EvaluationError, match="singular data"):
        _evaluate(problem, np.ones((3, problem.n_var)))

    tuning = HyperparameterTuningProblem()
    tuning._make_model = lambda params: _FailingModel(np.linalg.LinAlgError("not positive definite"))
    with pytest.raises(EvaluationError, match="not positive definite"):
        _evaluate(tuning, np.array([[0.5, 0.5, 0.0, 3.0], [0.1, 0.9, 1.0, 2.0]]))
