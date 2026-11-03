"""Benchmark execution through canonical durable studies."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vamos.experiment.benchmark.suites import BenchmarkExperiment, BenchmarkSuite
from vamos.experiment.study.errors import StudyOutputCollisionError
from vamos.experiment.study_analysis import SummarySource, derive_summary_rows
from vamos.study_artifacts import Study, StudyReport, StudySpec, StudySummary, create_study, load_study, plan_study

CONFIG_OVERRIDE_KEYS = ("population_size", "offspring_population_size", "max_evaluations", "seeds", "engine", "eval_strategy")


@dataclass(frozen=True, slots=True)
class BenchmarkStudy:
    """One homogeneous canonical study belonging to a benchmark suite."""

    experiment_index: int
    study: Study
    report: StudyReport
    summary: StudySummary


@dataclass(frozen=True, slots=True)
class _PlannedStudy:
    experiment_index: int
    spec: StudySpec
    destination: Path
    plan_id: str
    task_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    suite: BenchmarkSuite
    algorithms: tuple[str, ...]
    metrics: tuple[str, ...]
    base_output_dir: Path
    summary_path: Path | None
    studies: tuple[BenchmarkStudy, ...]

    @property
    def study_ids(self) -> tuple[str, ...]:
        return tuple(item.study.study_id for item in self.studies)

    @property
    def study_roots(self) -> tuple[Path, ...]:
        return tuple(item.study.root for item in self.studies)

    def summary_rows(self) -> tuple[dict[str, Any], ...]:
        """Render derived benchmark rows from canonical summaries."""
        sources = tuple(SummarySource(item.study.root, item.summary, {}) for item in self.studies)
        return derive_summary_rows(sources, indicators=self.metrics)


def run_benchmark_suite(
    suite: BenchmarkSuite,
    algorithms: Sequence[str] | None,
    metrics: Sequence[str] | None,
    base_output_dir: Path,
    global_config_overrides: Mapping[str, Any] | None = None,
) -> BenchmarkResult:
    """Run each benchmark experiment as one homogeneous canonical study.

    Every study is planned, and its destination checked, before the first one runs, so an
    invalid override or an occupied ``study-NNNN`` path stops the suite before any run.
    """
    algos = tuple(algorithms) if algorithms else tuple(suite.default_algorithms)
    metric_names = tuple(metrics) if metrics else tuple(suite.default_metrics)
    root = base_output_dir.absolute()
    overrides = validate_config_overrides(global_config_overrides or {})
    planned = tuple(
        _plan_experiment_study(suite, experiment, index, algos, root, overrides) for index, experiment in enumerate(suite.experiments)
    )
    executions = tuple(_run_planned_study(item) for item in planned)
    result = BenchmarkResult(suite, algos, metric_names, root, root / "summary" / "metrics.csv", executions)
    write_benchmark_summary(result)
    _write_suite_metadata(result, root / "summary" / "suite.json", overrides)
    return result


def validate_config_overrides(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``overrides`` after checking its keys and values; raise ``ValueError`` otherwise.

    The supported keys are ``CONFIG_OVERRIDE_KEYS``. ``max_evaluations`` and ``seeds`` replace the
    budget and the seeds of every experiment of the suite.
    """
    if not isinstance(overrides, Mapping):
        raise ValueError(f"benchmark config overrides must be a mapping, not {type(overrides).__name__}.")
    unknown = sorted(str(key) for key in overrides if key not in CONFIG_OVERRIDE_KEYS)
    if unknown:
        raise ValueError(f"unknown benchmark config keys: {', '.join(unknown)}. Supported keys: {', '.join(CONFIG_OVERRIDE_KEYS)}.")
    for field in ("population_size", "offspring_population_size", "max_evaluations"):
        _optional_int(overrides.get(field), field=field)
    _optional_seeds(overrides.get("seeds"))
    _optional_string(overrides.get("engine"), field="engine")
    _optional_string(overrides.get("eval_strategy"), field="eval_strategy")
    return dict(overrides)


def write_benchmark_summary(result: BenchmarkResult) -> Path | None:
    """Write ``result.summary_rows()`` to ``summary/metrics.csv``; return its path, or ``None`` without rows."""
    rows = result.summary_rows()
    if not rows:
        return None
    path = result.base_output_dir / "summary" / "metrics.csv"
    _write_summary(rows, path)
    return path


def load_benchmark_result(
    suite: BenchmarkSuite,
    *,
    algorithms: Sequence[str],
    metrics: Sequence[str],
    base_output_dir: Path,
) -> BenchmarkResult:
    """Regenerate benchmark views by loading only canonical study roots."""
    root = base_output_dir.absolute()
    studies: list[BenchmarkStudy] = []
    for manifest_path in root.rglob("study-manifest.json"):
        study = load_study(manifest_path.parent)
        labels = study.spec.labels or {}
        if labels.get("workflow") != "benchmark" or labels.get("suite") != suite.name:
            continue
        index = (study.spec.metadata or {}).get("experiment_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("Benchmark study metadata requires an integer experiment_index.")
        studies.append(BenchmarkStudy(index, study, study.inspect(), study.summarize()))
    studies.sort(key=lambda item: item.experiment_index)
    return BenchmarkResult(
        suite,
        tuple(algorithms),
        tuple(metrics),
        root,
        root / "summary" / "metrics.csv",
        tuple(studies),
    )


def _plan_experiment_study(
    suite: BenchmarkSuite,
    experiment: BenchmarkExperiment,
    index: int,
    algorithms: tuple[str, ...],
    root: Path,
    overrides: Mapping[str, Any],
) -> _PlannedStudy:
    problem_kwargs = {key: value for key, value in experiment.problem_params.items() if key not in {"n_var", "n_obj"}}
    spec = StudySpec(
        problems=[experiment.problem_name],
        algorithms=algorithms,
        seeds=_experiment_seeds(suite, experiment, overrides),
        max_evaluations=_experiment_budget(experiment, overrides),
        pop_size=_optional_int(overrides.get("population_size"), field="population_size"),
        engine=_optional_string(overrides.get("engine"), field="engine"),
        eval_strategy=str(overrides.get("eval_strategy", "serial")),
        n_var=_optional_int(experiment.problem_params.get("n_var"), field="n_var"),
        n_obj=_optional_int(experiment.problem_params.get("n_obj"), field="n_obj"),
        problem_kwargs=problem_kwargs,
        algorithm_configs=_algorithm_configs(algorithms, overrides),
        on_error="continue",
        labels={"workflow": "benchmark", "suite": suite.name},
        metadata={"experiment_index": index},
    )
    destination = root / f"study-{index:04d}"
    planned = plan_study(spec, output=destination)
    if planned.status != "ready":
        raise StudyOutputCollisionError(
            operation="run benchmark",
            reason="OUTPUT_COLLISION",
            path=destination,
            expected="destination path that does not exist",
            actual=planned.output.status,
            action="Choose another --output directory; no benchmark study was run. VAMOS never overwrites, reuses, or merges studies.",
        )
    return _PlannedStudy(index, spec, destination, planned.plan_id, planned.task_ids)


def _run_planned_study(planned: _PlannedStudy) -> BenchmarkStudy:
    created = create_study(planned.spec, output=planned.destination)
    if (planned.plan_id, planned.task_ids) != (created.plan_id, tuple(item.task_id for item in created.plan.tasks)):
        raise AssertionError("benchmark planning and creation identities differ")
    completed = created.run()
    return BenchmarkStudy(planned.experiment_index, completed, completed.inspect(), completed.summarize())


def _experiment_budget(experiment: BenchmarkExperiment, overrides: Mapping[str, Any]) -> int:
    budget = _optional_int(overrides.get("max_evaluations"), field="max_evaluations")
    if budget is not None:
        return budget
    return experiment.resolved_budget(population_size=_optional_int(overrides.get("population_size"), field="population_size"))


def _experiment_seeds(suite: BenchmarkSuite, experiment: BenchmarkExperiment, overrides: Mapping[str, Any]) -> list[int]:
    seeds = _optional_seeds(overrides.get("seeds"))
    if seeds is not None:
        return seeds
    return list(experiment.seeds or suite.default_seeds)


def _algorithm_configs(algorithms: tuple[str, ...], overrides: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    offspring = _optional_int(overrides.get("offspring_population_size"), field="offspring_population_size")
    if offspring is None:
        return {}
    return {algorithm: {"offspring_size": offspring} for algorithm in algorithms if algorithm.strip().lower() not in {"moead", "smpso"}}


def _write_summary(rows: tuple[dict[str, Any], ...], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_suite_metadata(result: BenchmarkResult, path: Path, overrides: Mapping[str, Any]) -> None:
    payload = {
        "suite": result.suite.name,
        "description": result.suite.description,
        "algorithms": list(result.algorithms),
        "metrics": list(result.metrics),
        "study_ids": list(result.study_ids),
        "plan_ids": [item.study.plan_id for item in result.studies],
        "config_overrides": dict(overrides),
        "experiments": [
            {
                "problem": experiment.problem_name,
                "params": experiment.problem_params,
                "evaluation_budget": experiment.evaluation_budget,
                "max_generations": experiment.max_generations,
                "max_evaluations": _experiment_budget(experiment, overrides),
                "seeds": _experiment_seeds(result.suite, experiment, overrides),
            }
            for experiment in result.suite.experiments
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _optional_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer when provided.")
    return value


def _optional_seeds(value: object) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in value):
        raise ValueError("seeds must be a non-empty list of non-negative integers when provided.")
    return list(value)


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string when provided.")
    return value


__all__ = [
    "CONFIG_OVERRIDE_KEYS",
    "BenchmarkResult",
    "BenchmarkStudy",
    "load_benchmark_result",
    "run_benchmark_suite",
    "validate_config_overrides",
    "write_benchmark_summary",
]
