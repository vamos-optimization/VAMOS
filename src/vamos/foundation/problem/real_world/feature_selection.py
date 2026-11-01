from __future__ import annotations

from typing import Any

import numpy as np

from ._training import TRAINING_ERRORS, report_training_failures


class FeatureSelectionProblem:
    """
    Binary feature-selection benchmark on a small classification dataset.

    Objectives (to minimize):
        f1: validation error (1 - accuracy)
        f2: number of selected features

    Requires scikit-learn; install the ``examples`` extras to enable.
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
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:  # pragma: no cover - exercised when sklearn missing
            raise ImportError("FeatureSelectionProblem requires scikit-learn. Install the 'examples' extras to enable it.") from exc

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
        scaler = StandardScaler()
        self._X_train = scaler.fit_transform(X_train)
        self._X_val = scaler.transform(X_val)
        self._y_train = y_train
        self._y_val = y_val

        self.n_var = data.data.shape[1]
        self.n_obj = 2
        self.encoding = "binary"
        self.xl = 0.0
        self.xu = 1.0

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError as exc:  # pragma: no cover - should be prevented by __init__
            raise ImportError("scikit-learn is required for evaluation.") from exc

        F = out["F"]
        failures: list[Exception] = []
        trained = 0
        for i, row in enumerate(X):
            mask = row > 0.5
            selected = int(np.sum(mask))
            if selected == 0:
                F[i, 0] = 1.0
                F[i, 1] = float(self.n_var)
                continue
            trained += 1
            try:
                clf = LogisticRegression(max_iter=200, solver="liblinear")
                clf.fit(self._X_train[:, mask], self._y_train)
                acc = clf.score(self._X_val[:, mask], self._y_val)
                err = 1.0 - float(acc)
            except TRAINING_ERRORS as exc:
                failures.append(exc)
                err = 1.0
            F[i, 0] = err
            F[i, 1] = float(selected)
        report_training_failures(type(self).__name__, failures, trained)


__all__ = ["FeatureSelectionProblem"]
