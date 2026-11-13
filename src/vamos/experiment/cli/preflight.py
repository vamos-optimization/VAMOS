from __future__ import annotations

import logging
from collections.abc import Iterable
from importlib.util import find_spec

from vamos.experiment.runtime.catalog import EXTERNAL_ALGORITHM_NAMES
from vamos.foundation.problem.resolver import PROBLEM_SET_PRESETS

_PROBLEM_EXTRA_REQUIREMENTS: dict[str, str] = {
    "ml_tuning": "examples",
    "fs_real": "examples",
}

_EXTRA_HINTS: dict[str, str] = {
    "analysis": 'pip install "vamos-optimization[analysis]"',
    "compute": 'pip install "vamos-optimization[compute]"',
    "examples": 'pip install "vamos-optimization[examples]"',
    "research": 'pip install "vamos-optimization[research]"',
}

_EXTERNAL_MODULES: dict[str, str] = {
    "pymoo": "pymoo",
    "jmetal": "jmetalpy",
    "pygmo": "pygmo",
}


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def _has_module(module: str) -> bool:
    return find_spec(module) is not None


def _warn(message: str) -> None:
    _logger().warning("Preflight: %s", message)


def _collect_problem_keys(args: object) -> Iterable[str]:
    problem_set = getattr(args, "problem_set", None)
    if problem_set:
        if problem_set == "all":
            keys: list[str] = []
            for name, problems in PROBLEM_SET_PRESETS.items():
                if name == "families":
                    continue
                keys.extend(problems)
            return keys
        return PROBLEM_SET_PRESETS.get(problem_set, ())
    problem = getattr(args, "problem", None)
    if problem:
        return (problem,)
    return ()


def _check_plotting(args: object) -> None:
    wants_plot = bool(getattr(args, "plot", False) or getattr(args, "live_viz", False))
    if wants_plot and not _has_module("matplotlib"):
        _warn(
            f"Plotting requested but matplotlib is not installed. Install with: {_EXTRA_HINTS['analysis']} or {_EXTRA_HINTS['examples']}."
        )


def _check_engine(args: object) -> None:
    engine = getattr(args, "engine", None)
    if engine == "numba" and not _has_module("numba"):
        _warn(f"Engine 'numba' selected but numba is not installed. Install with: {_EXTRA_HINTS['compute']}.")
    if engine == "moocore" and not _has_module("moocore"):
        _warn(f"Engine 'moocore' selected but moocore is not installed. Install with: {_EXTRA_HINTS['compute']}.")


def _check_external(args: object) -> None:
    algorithm = getattr(args, "algorithm", "")
    wants_external = bool(getattr(args, "include_external", False) or algorithm in EXTERNAL_ALGORITHM_NAMES)
    if not wants_external:
        return
    missing = []
    for module, label in _EXTERNAL_MODULES.items():
        if not _has_module(module):
            missing.append(label)
    if missing:
        missing_txt = ", ".join(missing)
        _warn(f"External baselines requested but missing: {missing_txt}. Install with: {_EXTRA_HINTS['research']}.")


def _check_problem_extras(args: object) -> None:
    keys = list(_collect_problem_keys(args))
    if not keys:
        return
    if any(_PROBLEM_EXTRA_REQUIREMENTS.get(key) == "examples" for key in keys):
        if not _has_module("sklearn"):
            _warn(f"Selected problem(s) require scikit-learn. Install with: {_EXTRA_HINTS['examples']}.")


def run_preflight_checks(args: object) -> None:
    """
    Emit friendly warnings for missing optional dependencies.
    """
    _check_plotting(args)
    _check_engine(args)
    _check_external(args)
    _check_problem_extras(args)


__all__ = ["run_preflight_checks"]
