from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ._training import TRAINING_ERRORS, report_training_failures


class HyperparameterTuningProblem:
    """
    Mixed-encoding hyperparameter tuning problem for a small scikit-learn classifier.

    Decision vector layout (n_var = 4):
        x0: scaled to C in [1e-2, 1e2] (continuous)
        x1: scaled to gamma in [1e-3, 1e1] (continuous)
        x2: kernel category in {rbf, linear, poly} (categorical via rounding)
        x3: scaled to polynomial degree in {2, 3, 4, 5} (integer, used for poly kernel)

    Objectives:
        f1: validation error (1 - accuracy) to minimize
        f2: model complexity proxy (support vectors / degree) to minimize

    This problem requires the optional ``scikit-learn`` dependency. Install the
    ``examples`` extras to enable it.
    """

    def __init__(
        self,
        dataset: str = "breast_cancer",
        test_size: float = 0.3,
        random_state: int = 0,
    ) -> None:
        try:
            from sklearn import datasets
            from sklearn.model_selection import train_test_split
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.svm import SVC
        except ImportError as exc:  # pragma: no cover - exercised only when sklearn is missing
            raise ImportError("HyperparameterTuningProblem requires scikit-learn. Install the 'examples' extras to enable it.") from exc

        if dataset == "breast_cancer":
            data = datasets.load_breast_cancer()
        elif dataset == "wine":
            data = datasets.load_wine()
        else:
            raise ValueError(f"Unsupported dataset '{dataset}'. Choose 'breast_cancer' or 'wine'.")

        X_train, X_val, y_train, y_val = train_test_split(
            data.data,
            data.target,
            test_size=test_size,
            random_state=random_state,
            stratify=data.target,
        )
        self._X_train = X_train
        self._X_val = X_val
        self._y_train = y_train
        self._y_val = y_val
        self._n_features = data.data.shape[1]
        self._make_model: Callable[[dict[str, Any]], Any] = lambda params: make_pipeline(StandardScaler(), SVC(**params))
        self._kernels = np.array(["rbf", "linear", "poly"])

        self.n_var = 4
        self.n_obj = 2
        self.encoding = "mixed"
        self.xl = np.array([0.0, 0.0, 0.0, 2.0], dtype=float)
        # kernel index upper bound is 2 (three kernels), polynomial degree in [2, 5]
        self.xu = np.array([1.0, 1.0, 2.0, 5.0], dtype=float)
        self.mixed_spec = {
            "real_idx": np.array([0, 1], dtype=int),
            "int_idx": np.array([3], dtype=int),
            "cat_idx": np.array([2], dtype=int),
            "real_lower": np.array([0.0, 0.0], dtype=float),
            "real_upper": np.array([1.0, 1.0], dtype=float),
            "int_lower": np.array([2], dtype=int),
            "int_upper": np.array([5], dtype=int),
            "cat_cardinality": np.array([len(self._kernels)], dtype=int),
        }

    def _decode_params(self, X: np.ndarray[Any, Any]) -> list[dict[str, Any]]:
        C = 10.0 ** (X[:, 0] * 4.0 - 2.0)  # 1e-2 .. 1e2
        gamma = 10.0 ** (X[:, 1] * 4.0 - 3.0)  # 1e-3 .. 1e1
        kernel_idx = np.clip(np.rint(X[:, 2]), 0, len(self._kernels) - 1).astype(int)
        kernels = self._kernels[kernel_idx]
        degrees = np.clip(np.rint(X[:, 3]), 2, 5).astype(int)

        params: list[dict[str, Any]] = []
        for c, g, k, d in zip(C, gamma, kernels, degrees):
            params.append({"C": float(c), "gamma": float(g), "kernel": str(k), "degree": int(d)})
        return params

    @staticmethod
    def _complexity(model: Any, params: dict[str, Any]) -> float:
        try:
            svc = model[-1]
        except (TypeError, IndexError, KeyError):  # not a pipeline
            svc = getattr(model, "steps", [[None, model]])[-1][-1]
        complexity = float(params.get("C", 1.0))
        if hasattr(svc, "n_support_"):
            complexity = float(np.sum(svc.n_support_))
        elif hasattr(svc, "support_vectors_"):
            complexity = float(svc.support_vectors_.shape[0])
        complexity += 0.1 * float(params.get("degree", 2))
        return complexity

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        params_list = self._decode_params(X)
        F = out["F"]
        failures: list[Exception] = []

        for i, params in enumerate(params_list):
            err = 1.0
            complexity = float(self._n_features)
            try:
                model = self._make_model(params)
                model.fit(self._X_train, self._y_train)
                acc = model.score(self._X_val, self._y_val)
                err = 1.0 - float(acc)
                complexity = self._complexity(model, params)
            except TRAINING_ERRORS as exc:
                failures.append(exc)
                err = 1.0
                complexity = float(self._n_features)
            F[i, 0] = err
            F[i, 1] = complexity
        report_training_failures(type(self).__name__, failures, len(params_list))


__all__ = ["HyperparameterTuningProblem"]
