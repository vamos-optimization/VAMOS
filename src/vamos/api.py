"""
User-facing API surface for VAMOS.

This module exports the same stable names as the top-level ``vamos`` package,
which examples import from: optimization, problem definition and selection,
results, and the run and study lifecycles. ``select_survivors``,
``EnvironmentalSelectionResult``, ``configure_logging`` and ``run_self_check``
are experimental. See ``docs/project/stability-and-versioning.md``.

The layered packages (``vamos.foundation``, ``vamos.engine``,
``vamos.experiment``, ``vamos.ux``) are internal implementation.
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import TYPE_CHECKING

from vamos.foundation.logging import configure_vamos_logging

if TYPE_CHECKING:
    from vamos.experiment.diagnostics.self_check import run_self_check
    from vamos.experiment.optimization_result import OptimizationResult, StudyResult
    from vamos.experiment.unified import optimize
    from vamos.foundation.problem.base import Problem
    from vamos.foundation.problem.builder import make_problem
    from vamos.foundation.problem.registry import (
        available_problem_names,
        make_problem_selection,
    )
    from vamos.run_artifacts import (
        CompatibilityReport,
        IncompleteRunMetadataError,
        LoadLimits,
        ReplayReport,
        RunManifest,
        StoredRun,
        VerificationReport,
        load_result,
        load_run,
        reproduce,
        save_result,
        verify_run,
    )
    from vamos.selection import EnvironmentalSelectionResult, select_survivors
    from vamos.study_artifacts import (
        Study,
        StudyLoadLimits,
        StudyPlanReport,
        StudyReport,
        StudySpec,
        StudySummary,
        create_study,
        load_study,
        plan_study,
    )

# The module of each public name. Names load on first access (PEP 562), so that importing any
# VAMOS module, which runs vamos/__init__.py, does not import the experiment layer and the
# algorithms; ``from vamos.api import optimize`` and ``vamos.api.optimize`` work as before.
_LAZY_NAMES: dict[str, str] = {
    "run_self_check": "vamos.experiment.diagnostics.self_check",
    "OptimizationResult": "vamos.experiment.optimization_result",
    "StudyResult": "vamos.experiment.optimization_result",
    "optimize": "vamos.experiment.unified",
    "Problem": "vamos.foundation.problem.base",
    "make_problem": "vamos.foundation.problem.builder",
    "available_problem_names": "vamos.foundation.problem.registry",
    "make_problem_selection": "vamos.foundation.problem.registry",
    "CompatibilityReport": "vamos.run_artifacts",
    "IncompleteRunMetadataError": "vamos.run_artifacts",
    "LoadLimits": "vamos.run_artifacts",
    "ReplayReport": "vamos.run_artifacts",
    "RunManifest": "vamos.run_artifacts",
    "StoredRun": "vamos.run_artifacts",
    "VerificationReport": "vamos.run_artifacts",
    "load_result": "vamos.run_artifacts",
    "load_run": "vamos.run_artifacts",
    "reproduce": "vamos.run_artifacts",
    "save_result": "vamos.run_artifacts",
    "verify_run": "vamos.run_artifacts",
    "EnvironmentalSelectionResult": "vamos.selection",
    "select_survivors": "vamos.selection",
    "Study": "vamos.study_artifacts",
    "StudyLoadLimits": "vamos.study_artifacts",
    "StudyPlanReport": "vamos.study_artifacts",
    "StudyReport": "vamos.study_artifacts",
    "StudySpec": "vamos.study_artifacts",
    "StudySummary": "vamos.study_artifacts",
    "create_study": "vamos.study_artifacts",
    "load_study": "vamos.study_artifacts",
    "plan_study": "vamos.study_artifacts",
}


def configure_logging(*, level: int = logging.INFO) -> None:
    """Configure a minimal console logger for VAMOS.

    Attaches a ``StreamHandler`` to the ``"vamos"`` logger if no handlers
    are already present.  This is intentionally opt-in: library code
    must never call ``logging.basicConfig()``.

    Experimental: not covered by the 1.x stability policy.

    Parameters
    ----------
    level : int, default ``logging.INFO``
        Logging level for the ``"vamos"`` logger.
    """
    configure_vamos_logging(level=level)


__all__ = [
    # Primary API
    "optimize",
    "select_survivors",
    "EnvironmentalSelectionResult",
    "Problem",
    "make_problem",
    "OptimizationResult",
    "StudyResult",
    "available_problem_names",
    "make_problem_selection",
    "run_self_check",
    "configure_logging",
    "save_result",
    "load_run",
    "load_result",
    "StoredRun",
    "RunManifest",
    "LoadLimits",
    "IncompleteRunMetadataError",
    "CompatibilityReport",
    "ReplayReport",
    "VerificationReport",
    "verify_run",
    "reproduce",
    "StudySpec",
    "Study",
    "StudyLoadLimits",
    "StudyPlanReport",
    "StudyReport",
    "StudySummary",
    "create_study",
    "load_study",
    "plan_study",
]


def __getattr__(name: str) -> object:
    module = _LAZY_NAMES.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
