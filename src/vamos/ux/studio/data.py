from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vamos.run_artifacts import load_run
from vamos.study_artifacts import load_study


@dataclass
class RunRecord:
    suite_name: str | None
    experiment_id: str
    problem_name: str
    algorithm_name: str
    seed: int
    fun: NDArray[Any]
    var: NDArray[Any] | None
    archive_fun: NDArray[Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    constraints: NDArray[Any] | None = None
    root: Path | None = None


@dataclass
class FrontRecord:
    problem_name: str
    algorithm_name: str
    points_F: NDArray[Any]
    points_X: NDArray[Any] | None
    constraints: NDArray[Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def load_run_from_directory(run_dir: Path) -> RunRecord:
    """Load one canonical v1 run through the data-only public reader."""
    run_dir = run_dir.resolve()
    stored = load_run(run_dir)
    result = stored.result  # The same verified read that ``load_result`` performs.
    manifest = stored.manifest
    resolved = manifest.resolved_spec
    archive = result.data.get("archive")
    archive_data = archive if isinstance(archive, Mapping) else {}
    labels = manifest.get("labels")
    suite_name = str(labels.get("suite")) if isinstance(labels, Mapping) and labels.get("suite") else None
    return RunRecord(
        suite_name=suite_name,
        experiment_id=manifest.run_id,
        problem_name=_component_name(resolved.get("problem")),
        algorithm_name=_component_name(resolved.get("algorithm")),
        seed=int(resolved["seed"]),
        fun=np.asarray(result.F),
        var=result.X,
        archive_fun=_array_or_none(archive_data.get("F")),
        metadata=manifest.as_dict(),
        constraints=_array_or_none(result.data.get("G")),
        root=run_dir,
    )


def load_runs_from_study(study_dir: Path, *, cache: dict[str, RunRecord] | None = None) -> list[RunRecord]:
    """Load selected successful runs through one canonical StudySummary.

    A canonical run never changes once published, so ``cache`` (keyed by the
    run's manifest path) lets repeated calls read only the runs added since.
    """
    study = load_study(study_dir)
    summary = study.summarize()
    runs: list[RunRecord] = []
    for row in summary.rows:
        if row.selected_run_id is None or row.run_manifest_path is None:
            continue
        run_dir = study.root / Path(row.run_manifest_path).parent
        key = str(run_dir)
        run = cache.get(key) if cache is not None else None
        if run is None:
            run = load_run_from_directory(run_dir)
            if cache is not None:
                cache[key] = run
        run.metadata["study_summary"] = row.as_dict()
        runs.append(run)
    return runs


def _array_or_none(value: object) -> NDArray[Any] | None:
    return value if isinstance(value, np.ndarray) else None


def _component_name(value: object) -> str:
    if isinstance(value, Mapping):
        resolution = value.get("resolution")
        if isinstance(resolution, Mapping) and isinstance(resolution.get("name"), str):
            return str(resolution["name"])
        component_id = value.get("component_id")
        if isinstance(component_id, str) and ":" in component_id:
            return component_id.split(":", 1)[1].split("@", 1)[0]
    return "unknown"


def build_fronts(
    runs: Iterable[RunRecord],
    *,
    problem_filter: str | None = None,
    merge_seeds: bool = True,
) -> list[FrontRecord]:
    """
    Build per-(problem, algorithm) fronts by optionally merging seeds.
    """
    grouped: dict[tuple[str, str], list[RunRecord]] = {}
    for run in runs:
        if problem_filter and run.problem_name != problem_filter:
            continue
        key = (
            (run.problem_name, run.algorithm_name)
            if merge_seeds
            else (
                f"{run.problem_name}_seed{run.seed}",
                run.algorithm_name,
            )
        )
        grouped.setdefault(key, []).append(run)

    fronts: list[FrontRecord] = []
    for (problem, algorithm), records in grouped.items():
        F = np.vstack([r.fun for r in records if r.fun.size])
        X = None
        if any(r.var is not None for r in records):
            X = np.vstack([r.var for r in records if r.var is not None])
        constraints = None
        with_points = [r for r in records if r.fun.size]
        if with_points and all(r.constraints is not None for r in with_points):
            constraints = np.vstack([r.constraints for r in with_points if r.constraints is not None])
        fronts.append(
            FrontRecord(
                problem_name=problem,
                algorithm_name=algorithm,
                points_F=F,
                points_X=X,
                constraints=constraints,
                extra={
                    "seeds": [r.seed for r in records],
                    "config": records[0].metadata.get("resolved_spec") if records else None,
                },
            )
        )
    return fronts


def normalize_objectives(F: NDArray[Any]) -> NDArray[np.float64]:
    F = np.asarray(F, dtype=float)
    mins = F.min(axis=0)
    maxs = F.max(axis=0)
    span = np.where(maxs - mins == 0, 1.0, maxs - mins)
    return np.asarray((F - mins) / span, dtype=float)
