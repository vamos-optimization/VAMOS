from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from vamos.foundation.exceptions import _suggest_names
from vamos.foundation.problem.registry import ProblemSelection

from .jmetalpy import _run_jmetalpy_nsga2, _run_jmetalpy_perm_nsga2
from .pygmo import _run_pygmo_nsga2
from .pymoo import _run_pymoo_nsga2, _run_pymoo_perm_nsga2

_EXTERNAL_ALGORITHM_ADAPTERS: dict[str, ExternalAlgorithmAdapter] | None = None
_EXTERNAL_DOCS = "docs/reference/algorithms.md"
_TROUBLESHOOTING_DOCS = "docs/guide/troubleshooting.md"


def _format_unknown_external(name: str, options: list[str]) -> str:
    parts = [f"Unknown external algorithm '{name}'.", f"Available: {', '.join(options)}."]
    suggestions = _suggest_names(name, options)
    if suggestions:
        if len(suggestions) == 1:
            parts.append(f"Did you mean '{suggestions[0]}'?")
        else:
            parts.append("Did you mean one of: " + ", ".join(f"'{item}'" for item in suggestions) + "?")
    parts.append(f"Docs: {_EXTERNAL_DOCS}.")
    parts.append(f"Troubleshooting: {_TROUBLESHOOTING_DOCS}.")
    return " ".join(parts)


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class ExternalAlgorithmAdapter:
    """
    Thin adapter to standardize external baseline invocation.
    """

    def __init__(self, name: str, runner_fn: Callable[..., Any]) -> None:
        self.name = name
        self._runner_fn = runner_fn

    def run(
        self,
        selection: ProblemSelection,
        *,
        use_native_problem: bool,
        config: Any,
        make_metrics: Callable[..., Any],
        print_banner: Callable[..., Any],
        print_results: Callable[..., Any],
    ) -> Any:
        return self._runner_fn(
            selection,
            use_native_problem=use_native_problem,
            config=config,
            make_metrics=make_metrics,
            print_banner=print_banner,
            print_results=print_results,
        )


EXTERNAL_ALGORITHM_RUNNERS: dict[str, Callable[..., object]] = {
    "pymoo_nsga2": _run_pymoo_nsga2,
    "jmetalpy_nsga2": _run_jmetalpy_nsga2,
    "pygmo_nsga2": _run_pygmo_nsga2,
    "pymoo_perm_nsga2": _run_pymoo_perm_nsga2,
    "jmetalpy_perm_nsga2": _run_jmetalpy_perm_nsga2,
}


def _get_external_algorithm_adapters() -> dict[str, ExternalAlgorithmAdapter]:
    global _EXTERNAL_ALGORITHM_ADAPTERS
    if _EXTERNAL_ALGORITHM_ADAPTERS is None:
        _EXTERNAL_ALGORITHM_ADAPTERS = {name: ExternalAlgorithmAdapter(name, fn) for name, fn in EXTERNAL_ALGORITHM_RUNNERS.items()}
    return _EXTERNAL_ALGORITHM_ADAPTERS


def resolve_external_algorithm(name: str) -> ExternalAlgorithmAdapter:
    adapters = _get_external_algorithm_adapters()
    try:
        return adapters[name]
    except KeyError as exc:  # pragma: no cover - defensive guard
        available = sorted(adapters)
        raise ValueError(_format_unknown_external(name, available)) from exc


def run_external(
    name: str,
    selection: ProblemSelection,
    *,
    use_native_problem: bool,
    config: Any,
    make_metrics: Callable[..., Any],
    print_banner: Callable[..., Any],
    print_results: Callable[..., Any],
) -> Any:
    adapter = resolve_external_algorithm(name)
    try:
        return adapter.run(
            selection,
            use_native_problem=use_native_problem,
            config=config,
            make_metrics=make_metrics,
            print_banner=print_banner,
            print_results=print_results,
        )
    except ImportError as exc:
        _logger().warning("Skipping %s: %s", name, exc)
        _logger().info("%s", "=" * 80)
        return None
