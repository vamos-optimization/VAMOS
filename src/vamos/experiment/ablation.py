"""Canonical durable-study execution for ablation plans."""

from __future__ import annotations

import csv
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from vamos.engine.config.spec import _ALGORITHM_BLOCKS
from vamos.engine.config.variation import ensure_operator_tuple
from vamos.engine.tuning.ablation import AblationPlan, AblationTask, AblationVariant
from vamos.experiment._execution_support import VariationConfigs
from vamos.experiment.study.writing import fsync_directory
from vamos.experiment.study_analysis import SummarySource, derive_summary_rows
from vamos.foundation.exceptions import ConfigurationError, _suggest_names
from vamos.study_artifacts import Study, StudyPlanReport, StudyReport, StudySpec, StudySummary, create_study, plan_study

# Settings of the ablation itself; every other key of a variant's configuration
# is a field of the algorithm configuration.
_ABLATION_KEYS = {"population_size", "offspring_population_size", "eval_strategy", "output_root"}
_OPERATOR_FIELDS = {"crossover", "mutation", "selection", "repair", "aggregation"}


@dataclass(frozen=True, slots=True)
class AblationStudy:
    """One homogeneous canonical study within an ablation execution."""

    variant: str
    problem: str
    study: Study
    report: StudyReport
    summary: StudySummary


@dataclass(frozen=True, slots=True)
class AblationResult:
    """Ablation-specific grouping of canonical study projections."""

    studies: tuple[AblationStudy, ...]

    @property
    def study_ids(self) -> tuple[str, ...]:
        return tuple(item.study.study_id for item in self.studies)

    @property
    def study_roots(self) -> tuple[Path, ...]:
        return tuple(item.study.root for item in self.studies)

    def summary_rows(self) -> tuple[dict[str, Any], ...]:
        """Return derived rows while retaining canonical task/run evidence."""
        sources = tuple(SummarySource(item.study.root, item.summary, {"variant": item.variant}) for item in self.studies)
        return derive_summary_rows(sources, indicators=("hv",))


def write_ablation_csv(result: AblationResult, path: str | Path) -> Path:
    """Atomically write an explicit, regenerable table derived from summaries."""
    destination = Path(path)
    if os.path.lexists(destination):
        raise FileExistsError(f"Derived ablation output already exists: {destination}")
    rows = result.summary_rows()
    if not rows:
        raise ValueError("Cannot write an ablation table without summary rows.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        fieldnames = tuple(dict.fromkeys(key for row in rows for key in row))
        with temporary.open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"Derived ablation output already exists: {destination}") from exc
        fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def run_ablation_plan(
    plan: AblationPlan,
    *,
    algorithm: str,
    output: str | Path,
    base_config: Mapping[str, Any] | None = None,
    variations_by_variant: Mapping[str, VariationConfigs] | None = None,
    engine: str | None = None,
) -> AblationResult:
    """Execute each scientifically homogeneous ablation group as a study.

    Every key of ``base_config`` and of a variant's ``config_overrides`` must be
    an ablation setting (``population_size``, ``offspring_population_size``,
    ``eval_strategy`` or ``output_root``) or a field of the algorithm
    configuration, which then reaches the study's ``algorithm_configs``. Any
    other key raises :class:`~vamos.exceptions.ConfigurationError`. Each study
    is written to ``<output_root>/study-NNNN``, where ``output_root`` is the
    variant's setting, or ``output`` when it has none. Every study is planned
    before the first one runs.
    """
    groups = _homogeneous_groups(plan)
    root = Path(output).absolute()
    planned_groups: list[tuple[AblationTask, StudySpec, Path, StudyPlanReport]] = []
    for index, tasks in enumerate(groups):
        spec = _study_spec(
            plan,
            tasks,
            algorithm=algorithm,
            base_config=base_config,
            variations_by_variant=variations_by_variant,
            engine=engine,
        )
        variant_root = tasks[0].variant.apply(base_config).get("output_root")
        destination = (Path(str(variant_root)).absolute() if variant_root else root) / f"study-{index:04d}"
        planned_groups.append((tasks[0], spec, destination, plan_study(spec, output=destination)))

    executions: list[AblationStudy] = []
    for first, spec, destination, planned in planned_groups:
        created = create_study(spec, output=destination)
        if (planned.plan_id, planned.task_ids) != (created.plan_id, tuple(item.task_id for item in created.plan.tasks)):
            raise AssertionError("ablation planning and creation identities differ")
        completed = created.run()
        executions.append(
            AblationStudy(
                variant=first.variant.name,
                problem=first.problem,
                study=completed,
                report=completed.inspect(),
                summary=completed.summarize(),
            )
        )
    return AblationResult(tuple(executions))


def _homogeneous_groups(plan: AblationPlan) -> tuple[tuple[AblationTask, ...], ...]:
    groups: list[list[AblationTask]] = []
    keys: list[tuple[str, str, int, str | None]] = []
    for task in plan.tasks:
        key = (task.problem, task.variant.name, task.max_evals, task.engine or plan.engine)
        if key not in keys:
            keys.append(key)
            groups.append([])
        groups[keys.index(key)].append(task)
    return tuple(tuple(group) for group in groups)


def _study_spec(
    plan: AblationPlan,
    tasks: tuple[AblationTask, ...],
    *,
    algorithm: str,
    base_config: Mapping[str, Any] | None,
    variations_by_variant: Mapping[str, VariationConfigs] | None,
    engine: str | None,
) -> StudySpec:
    first = tasks[0]
    config = first.variant.apply(base_config)
    variation = variations_by_variant.get(first.variant.name) if variations_by_variant is not None else None
    algorithm_config = _algorithm_config(config, variation, algorithm, variant=first.variant)
    return StudySpec(
        problems=[first.problem],
        algorithms=[algorithm],
        seeds=[task.seed for task in tasks],
        max_evaluations=first.max_evals,
        pop_size=_optional_int(config.get("population_size"), field="population_size"),
        engine=engine or first.engine or plan.engine,
        eval_strategy=str(config.get("eval_strategy", "serial")),
        algorithm_configs={algorithm: algorithm_config},
        on_error="fail_fast",
        labels={"workflow": "ablation", "variant": first.variant.name},
        metadata={
            "ablation": plan.metadata,
            "variant": {
                "name": first.variant.name,
                "label": first.variant.label or first.variant.name,
                "tags": list(first.variant.tags),
            },
        },
    )


def _algorithm_config(
    config: Mapping[str, Any],
    variation: VariationConfigs | None,
    algorithm: str,
    *,
    variant: AblationVariant,
) -> dict[str, Any]:
    key = algorithm.strip().lower()
    result: dict[str, Any] = {}
    offspring = config.get("offspring_population_size")
    if offspring is not None and key not in {"moead", "smpso"}:
        result["offspring_size"] = _optional_int(offspring, field="offspring_population_size")
    known = _config_fields(key)
    for name, value in config.items():
        if name in _ABLATION_KEYS:
            continue
        if known is not None and name not in known:
            origin = f"variant '{variant.name}' config_overrides" if name in variant.config_overrides else "base_config"
            raise _unknown_field_error(name, origin=origin, algorithm=key, known=known)
        result[name] = value
    if variation is not None:
        selected = variation.copy_field(key)
        if selected is not None:
            result.update(selected)
    # The study planner reads operators as (name, params) pairs; the mapping form
    # {method: ..., <params>} of the CLI configuration is accepted too.
    for name in _OPERATOR_FIELDS & result.keys():
        if isinstance(result[name], Mapping):
            result[name] = ensure_operator_tuple(result[name], key=name)
    return result


def _config_fields(algorithm: str) -> frozenset[str] | None:
    """Fields of a built-in algorithm configuration, or ``None`` for a plugin (the planner checks those)."""
    config_type = _ALGORITHM_BLOCKS.get(algorithm)
    return None if config_type is None else frozenset(field.name for field in fields(config_type))


def _unknown_field_error(name: str, *, origin: str, algorithm: str, known: frozenset[str]) -> ConfigurationError:
    matches = _suggest_names(name, sorted(known | _ABLATION_KEYS))
    if matches:
        suggestion = f"Did you mean {' or '.join(repr(match) for match in matches)}?"
    else:
        suggestion = f"Known fields: {', '.join(sorted(known))}."
    if name.split("_", 1)[0] in _OPERATOR_FIELDS:
        suggestion += " Operator parameters go inside the operator, for example crossover: [sbx, {prob: 0.5}]."
    return ConfigurationError(
        f"Ablation {origin} sets '{name}', which is neither an ablation setting "
        f"({', '.join(sorted(_ABLATION_KEYS))}) nor a field of the {algorithm} configuration.",
        suggestion=suggestion,
    )


def _optional_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer when provided.")
    return value


__all__ = ["AblationResult", "AblationStudy", "run_ablation_plan", "write_ablation_csv"]
