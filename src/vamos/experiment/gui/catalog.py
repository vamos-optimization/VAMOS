"""Read-only discovery of canonical runs for the explorer view.

Discovery walks a directory tree for ``manifest.json`` files and summarizes each
run through the canonical, data-only ``load_run`` reader; arrays are loaded only
for the run the user opens. Unreadable runs are reported, not hidden.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

MANIFEST_FILENAME = "manifest.json"
DEFAULT_MAX_DEPTH = 6
DEFAULT_LIMIT = 500
_SKIPPED_DIRS = ("node_modules", "site-packages")


@dataclass(frozen=True)
class RunSummary:
    """One row of the explorer table."""

    path: Path
    relative_path: str
    status: str
    problem: str
    algorithm: str
    n_objectives: int | None = None
    n_solutions: int | None = None
    evaluations: int | None = None
    seed: int | None = None
    started_at: str = ""
    error: str | None = None

    @property
    def loadable(self) -> bool:
        return self.status == "succeeded" and self.error is None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _int(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _component_name(component: Mapping[str, Any]) -> str:
    component_id = component.get("component_id")
    if isinstance(component_id, str) and ":" in component_id:
        return component_id.split(":", 1)[1].split("@", 1)[0]
    return "?"


def summarize_run(run_dir: str | Path, root: str | Path) -> RunSummary:
    """Summarize one stored run without materializing its arrays."""
    from vamos.experiment.artifacts import load_run

    path = Path(run_dir)
    try:
        relative = path.relative_to(Path(root)).as_posix()
    except ValueError:
        relative = path.as_posix()
    try:
        stored = load_run(path)
    except Exception as exc:  # noqa: BLE001 - every unreadable run is listed with its reason
        return RunSummary(
            path=path, relative_path=relative, status="unreadable", problem="?", algorithm="?", error=f"{type(exc).__name__}: {exc}"
        )
    manifest = stored.manifest
    resolved = _mapping(manifest.resolved_spec)
    outcome = _mapping(manifest.get("outcome"))
    timestamps = _mapping(manifest.get("timestamps"))
    started = timestamps.get("started_at")
    return RunSummary(
        path=path,
        relative_path=relative,
        status=str(stored.status),
        problem=_component_name(_mapping(resolved.get("problem"))),
        algorithm=_component_name(_mapping(resolved.get("algorithm"))),
        n_objectives=_int(outcome.get("n_objectives")),
        n_solutions=_int(outcome.get("n_solutions")),
        evaluations=_int(outcome.get("evaluations")),
        seed=_int(resolved.get("seed")),
        started_at=started if isinstance(started, str) else "",
    )


def discover_runs(root: str | Path, *, max_depth: int = DEFAULT_MAX_DEPTH, limit: int = DEFAULT_LIMIT) -> list[RunSummary]:
    """Return summaries of runs under ``root``, newest first.

    Hidden and staging directories (names starting with ``.``) are skipped and a
    run directory is not searched further. At most ``limit`` runs are returned.
    """
    base = Path(root)
    if not base.is_dir():
        return []
    found: list[RunSummary] = []
    for current, dirs, files in os.walk(base):
        depth = len(Path(current).relative_to(base).parts)
        if MANIFEST_FILENAME in files:
            found.append(summarize_run(current, base))
            dirs[:] = []
            if len(found) >= limit:
                break
            continue
        if depth >= max_depth:
            dirs[:] = []
            continue
        dirs[:] = sorted(name for name in dirs if not name.startswith((".", "__")) and name not in _SKIPPED_DIRS)
    return sorted(found, key=lambda item: item.started_at, reverse=True)


def load_front(run_dir: str | Path) -> tuple[npt.NDArray[np.float64], npt.NDArray[Any] | None]:
    """Return ``(F, X)`` of a stored run through the canonical result reader."""
    from vamos.experiment.artifacts import load_result

    result = load_result(Path(run_dir))
    F = np.asarray(result.F, dtype=float)
    X = None if result.X is None else np.asarray(result.X)
    return F, X


__all__ = ["DEFAULT_LIMIT", "DEFAULT_MAX_DEPTH", "MANIFEST_FILENAME", "RunSummary", "discover_runs", "load_front", "summarize_run"]
