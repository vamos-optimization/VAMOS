"""Model training failures in the machine-learning benchmark problems."""

from __future__ import annotations

import warnings
from collections import Counter

import numpy as np

from vamos.foundation.exceptions import EvaluationError

# Failures a model can raise on legitimate data (degenerate folds, singular matrices). Anything
# else is a bug in the model or the problem and propagates.
TRAINING_ERRORS: tuple[type[Exception], ...] = (ValueError, np.linalg.LinAlgError)


def report_training_failures(problem: str, failures: list[Exception], n_models: int) -> None:
    """Warn about the models of an evaluation that failed to train, or raise if all of them did.

    The report names the exception types and their counts, never their text: it comes from the
    model library and can hold data or credentials that must not reach logs or run records.
    """
    if not failures:
        return
    counts = Counter(type(failure).__name__ for failure in failures)
    detail = ", ".join(f"{count} {name}" for name, count in sorted(counts.items()))
    if len(failures) == n_models:
        raise EvaluationError(f"{problem}: every model of the evaluation failed to train ({detail}).")
    warnings.warn(
        f"{problem}: {len(failures)} of {n_models} models failed to train ({detail}); their error objective is 1.0.",
        RuntimeWarning,
        stacklevel=3,
    )


__all__ = ["TRAINING_ERRORS", "report_training_failures"]
