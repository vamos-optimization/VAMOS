"""
Public exceptions namespace.

Re-exports the canonical exception classes from vamos.foundation.exceptions, and the base
errors of the run-artifact and study APIs. All of them are ``VAMOSError``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from .foundation.exceptions import (
    BackendNotAvailableError,
    BoundsError,
    ConfigurationError,
    ConstraintViolationError,
    ConvergenceError,
    DataError,
    DependencyError,
    EvaluationError,
    InvalidAlgorithmError,
    InvalidEngineError,
    InvalidOperatorError,
    InvalidProblemError,
    InvalidResultsError,
    MissingConfigError,
    NoSolutionsError,
    OptimizationError,
    ProblemDimensionError,
    ProblemError,
    ResultSelectionError,
    ResultsNotFoundError,
    VAMOSError,
)

if TYPE_CHECKING:
    from .experiment.artifacts.errors import RunArtifactError
    from .experiment.study.errors import StudyError

# The run-artifact and study errors belong to the experiment layer: they load on first access,
# so that importing this namespace does not import that layer.
_EXPERIMENT_ERRORS = {
    "RunArtifactError": "vamos.experiment.artifacts.errors",
    "StudyError": "vamos.experiment.study.errors",
}

__all__ = [
    "VAMOSError",
    "ConfigurationError",
    "InvalidAlgorithmError",
    "InvalidEngineError",
    "InvalidOperatorError",
    "MissingConfigError",
    "ProblemError",
    "InvalidProblemError",
    "ProblemDimensionError",
    "BoundsError",
    "OptimizationError",
    "ConvergenceError",
    "EvaluationError",
    "ConstraintViolationError",
    "DataError",
    "ResultsNotFoundError",
    "InvalidResultsError",
    "ResultSelectionError",
    "NoSolutionsError",
    "DependencyError",
    "BackendNotAvailableError",
    "RunArtifactError",
    "StudyError",
]


def __getattr__(name: str) -> object:
    module = _EXPERIMENT_ERRORS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
