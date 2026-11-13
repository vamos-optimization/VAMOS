"""
VAMOS package facade.

Stable facades (see ``docs/project/stability-and-versioning.md``):
- `vamos` (and `vamos.api`, which exports the same names) for optimization,
  problems, results, and the run and study lifecycles.
- `vamos.algorithms` for algorithm configurations and identifiers.
- `vamos.problems` for curated problem classes.
- `vamos.run_artifacts` and `vamos.study_artifacts` for the run and study
  lifecycles alone.

`vamos.ux.api` (analysis and visualization) is experimental, as are
`select_survivors`, `EnvironmentalSelectionResult`, `configure_logging` and
`run_self_check` exported here.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from vamos.foundation.version import get_version as _get_version

if TYPE_CHECKING:
    from vamos.api import (
        CompatibilityReport,
        EnvironmentalSelectionResult,
        IncompleteRunMetadataError,
        LoadLimits,
        OptimizationResult,
        Problem,
        ReplayReport,
        RunManifest,
        StoredRun,
        Study,
        StudyLoadLimits,
        StudyPlanReport,
        StudyReport,
        StudyResult,
        StudySpec,
        StudySummary,
        VerificationReport,
        available_problem_names,
        configure_logging,
        create_study,
        load_result,
        load_run,
        load_study,
        make_problem,
        make_problem_selection,
        optimize,
        plan_study,
        reproduce,
        run_self_check,
        save_result,
        select_survivors,
        verify_run,
    )

__all__ = [
    "__version__",
    # Optimization
    "optimize",
    "select_survivors",
    "EnvironmentalSelectionResult",
    "Problem",
    "make_problem",
    "OptimizationResult",
    "StudyResult",
    "configure_logging",
    "available_problem_names",
    "make_problem_selection",
    "run_self_check",
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
    "problems",
]


def __getattr__(name: str) -> object:
    if name == "__version__":
        return _get_version()
    if name == "problems":
        return importlib.import_module(".problems", __name__)
    if name in __all__:
        # The facade names load from vamos.api on first access, so that importing any VAMOS
        # module does not import the experiment layer and the algorithms.
        value = getattr(importlib.import_module("vamos.api"), name)
        globals()[name] = value
        return value
    if not name.startswith("__"):
        # Subpackages such as vamos.api or vamos.run_artifacts, which the eager facade used to
        # import, stay reachable as attributes after `import vamos`.
        try:
            return importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as exc:
            if exc.name != f"{__name__}.{name}":
                raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
