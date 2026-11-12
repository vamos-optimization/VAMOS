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

from vamos.experiment.diagnostics.self_check import run_self_check
from vamos.experiment.optimization_result import OptimizationResult, StudyResult

# Unified API - the primary entry point
from vamos.experiment.unified import optimize
from vamos.foundation.logging import configure_vamos_logging
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
