"""Deterministic, execution-free StudySpec resolution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from math import comb
from typing import Any

from vamos.engine.algorithm.config.base import ReferenceDirectionPopulationError
from vamos.experiment.artifacts.models import deep_freeze, deep_thaw
from vamos.experiment.artifacts.resolved_reconstruction import is_reconstructable_evaluation_backend
from vamos.experiment.artifacts.specs import RunSpecInputs, build_run_specs, is_registry_problem
from vamos.experiment.auto import _compute_max_evaluations, _compute_pop_size, _resolve_problem, _select_algorithm
from vamos.experiment.optimize import _build_algorithm_config
from vamos.experiment.runtime.catalog import resolve_engine_details
from vamos.foundation.encoding import normalize_encoding

from .errors import DuplicateStudyTaskError, InvalidStudySpecError, StudyError, UnresolvedStudyTaskError
from .identity import compute_plan_id, compute_task_digest, compute_task_id
from .models import SCHEMA_VERSION, PlanTask, ResolvedStudyPlan, StudySpec
from .serialization import document_self_hash


@dataclass(frozen=True, slots=True)
class _Candidate:
    index: int
    problem: str
    algorithm: str
    seed: int


def resolve_spec(spec: StudySpec) -> ResolvedStudyPlan:
    """Resolve every matrix candidate and freeze one immutable sorted plan."""
    _reject_negative_seeds(spec.seeds)
    candidates = (
        _Candidate(index, problem, algorithm, seed)
        for index, (problem, algorithm, seed) in enumerate(product(spec.problems, spec.algorithms, spec.seeds))
    )
    tasks = tuple(_resolve_candidate(spec, candidate) for candidate in candidates)
    _reject_duplicates(tasks)
    sorted_tasks = tuple(sorted(tasks, key=lambda task: task.task_id))
    plan_id = compute_plan_id(task.task_id for task in sorted_tasks)
    provisional = ResolvedStudyPlan(plan_id=plan_id, tasks=sorted_tasks, document_sha256="")
    return ResolvedStudyPlan(plan_id=plan_id, tasks=sorted_tasks, document_sha256=_plan_hash(provisional))


def _reject_negative_seeds(seeds: Sequence[int]) -> None:
    # Validated here rather than in StudySpec so that studies published before this
    # check still load; their negative-seed tasks could never run.
    for index, seed in enumerate(seeds):
        if seed < 0:
            raise InvalidStudySpecError(
                operation="resolve study plan",
                reason="INVALID_STUDY_SPEC",
                field=f"seeds[{index}]",
                expected="non-negative integer seed",
                actual=seed,
                action="Use seeds >= 0; no study directory was published.",
            )


def _resolve_candidate(spec: StudySpec, candidate: _Candidate) -> PlanTask:
    try:
        problem = _resolve_problem(
            candidate.problem,
            n_var=spec.n_var,
            n_obj=spec.n_obj,
            problem_kwargs=deep_thaw(spec.problem_kwargs),
        )
        n_var = int(problem.n_var)
        n_obj = int(problem.n_obj)
        encoding = normalize_encoding(getattr(problem, "encoding", "real"))
        algorithm = _select_algorithm(n_obj, encoding) if candidate.algorithm == "auto" else candidate.algorithm
        population_size = spec.pop_size or _compute_pop_size(n_var, n_obj)
        budget = spec.max_evaluations or _compute_max_evaluations(n_var, n_obj)
        engine, engine_source = resolve_engine_details(spec.engine, algorithm=algorithm)
        try:
            config = _resolved_algorithm_config(spec, candidate.algorithm, algorithm, population_size, n_var, n_obj, encoding)
        except ReferenceDirectionPopulationError as exc:
            raise UnresolvedStudyTaskError(
                operation="resolve study plan",
                reason="REFERENCE_DIRECTION_POPULATION_MISMATCH",
                field=f"$.matrix[{candidate.index}].pop_size",
                expected=f"a population that fits a complete reference-direction lattice for n_obj={n_obj}",
                actual=population_size,
                action=exc.message,
            ) from exc
        resolved_population_size = _validate_execution_shape(
            algorithm=algorithm,
            config=config,
            budget=budget,
            n_obj=n_obj,
            candidate=candidate,
        )
        requested, resolved = build_run_specs(
            RunSpecInputs(
                problem_built_in=is_registry_problem(candidate.problem, problem, candidate.problem),
                problem_label=candidate.problem,
                problem_kwargs=deep_thaw(spec.problem_kwargs),
                n_var_requested=spec.n_var,
                n_obj_requested=spec.n_obj,
                n_var=n_var,
                n_obj=n_obj,
                encoding=encoding,
                algorithm_requested=candidate.algorithm,
                algorithm=algorithm,
                algorithm_config=config,
                algorithm_config_explicit=bool(_algorithm_overrides(spec, candidate.algorithm, algorithm)),
                max_evaluations_requested=spec.max_evaluations,
                termination=("max_evaluations", budget),
                pop_size_requested=spec.pop_size,
                resolved_pop_size=resolved_population_size,
                engine_requested=spec.engine,
                engine=engine,
                eval_strategy=spec.eval_strategy,
                seed_requested=candidate.seed,
                seed=candidate.seed,
                default_sources={
                    "algorithm": "auto" if candidate.algorithm == "auto" else "explicit",
                    "pop_size": "explicit" if spec.pop_size is not None else "auto",
                    "max_evaluations": "explicit" if spec.max_evaluations is not None else "auto",
                    "engine": engine_source,
                    "algorithm_config": "explicit" if _algorithm_overrides(spec, candidate.algorithm, algorithm) else "auto",
                },
            )
        )
        _validate_resolved_components(resolved, candidate=candidate)
    except StudyError:
        raise
    except Exception as exc:
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="UNRESOLVED_TASK",
            field=f"$.matrix[{candidate.index}]",
            expected="supported problem, algorithm, backend, and complete JSON configuration",
            actual={
                "problem": candidate.problem,
                "algorithm": candidate.algorithm,
                "seed": candidate.seed,
                "error": type(exc).__name__,
            },
            action="Correct this matrix candidate; no study directory was published.",
        ) from exc
    task_id = compute_task_id(resolved)
    digest = compute_task_digest(resolved)
    return PlanTask(
        plan_index=candidate.index,
        requested_run=deep_freeze(requested),
        resolved_run_spec=deep_freeze(resolved),
        task_id=task_id,
        task_spec_sha256=digest,
    )


def _resolved_algorithm_config(
    spec: StudySpec,
    requested_algorithm: str,
    algorithm: str,
    population_size: int,
    n_var: int,
    n_obj: int,
    encoding: str,
) -> dict[str, Any]:
    overrides = _algorithm_overrides(spec, requested_algorithm, algorithm)
    config_pop_size: int | None = population_size
    if spec.pop_size is None and algorithm.lower() in {"moead", "nsgaiii", "rvea"}:
        config_pop_size = None
    # A per-algorithm population replaces the shared one before the default derives its lattice from it.
    override_pop_size = overrides.get("pop_size")
    if isinstance(override_pop_size, int) and not isinstance(override_pop_size, bool):
        config_pop_size = override_pop_size
    built = _build_algorithm_config(
        algorithm,
        pop_size=config_pop_size,
        n_var=n_var,
        n_obj=n_obj,
        encoding=encoding,
    )
    config = dict(built.to_dict())
    unknown = sorted(set(overrides) - set(config))
    if unknown:
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="UNRESOLVED_TASK",
            field=f"$.algorithm_configs.{algorithm}",
            expected=f"known configuration fields: {sorted(config)}",
            actual={"unknown": unknown},
            action="Remove unknown algorithm configuration fields; no study directory was published.",
        )
    config.update(overrides)
    return config


def _algorithm_overrides(spec: StudySpec, requested: str, resolved: str) -> dict[str, Any]:
    configs = spec.algorithm_configs or {}
    raw = configs.get(requested, configs.get(resolved, {}))
    if not isinstance(raw, Mapping):
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="UNRESOLVED_TASK",
            field=f"$.algorithm_configs.{requested}",
            expected="JSON object",
            actual=type(raw).__name__,
            action="Use an object keyed by supported algorithm configuration fields.",
        )
    thawed = deep_thaw(raw)
    if not isinstance(thawed, dict):
        raise AssertionError("algorithm override is not an object")
    return thawed


def _validate_execution_shape(
    *,
    algorithm: str,
    config: Mapping[str, Any],
    budget: int,
    n_obj: int,
    candidate: _Candidate,
) -> int:
    population = config.get("pop_size")
    if isinstance(population, bool) or not isinstance(population, int) or population < 1:
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="INVALID_POPULATION_SIZE",
            field=f"$.matrix[{candidate.index}].pop_size",
            expected="positive resolved initial population size",
            actual=population,
            action="Correct the population configuration; no study directory was published.",
        )
    if budget < population:
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="INVALID_EVALUATION_BUDGET",
            field=f"$.matrix[{candidate.index}].max_evaluations",
            expected=f"integer >= resolved initial population size ({population})",
            actual=budget,
            action="Increase max_evaluations or reduce the population; no study directory was published.",
        )
    if algorithm.lower() == "nsgaiii":
        _validate_nsgaiii_reference_directions(config, population=population, n_obj=n_obj, candidate=candidate)
    if algorithm.lower() == "rvea":
        _validate_rvea_reference_directions(config, population=population, n_obj=n_obj, candidate=candidate)
    return population


def _validate_nsgaiii_reference_directions(config: Mapping[str, Any], *, population: int, n_obj: int, candidate: _Candidate) -> None:
    raw = config.get("reference_directions")
    if not isinstance(raw, Mapping):
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="INVALID_REFERENCE_DIRECTIONS",
            field=f"$.matrix[{candidate.index}].algorithm_config.reference_directions",
            expected="object with path=null and positive integer divisions",
            actual=type(raw).__name__,
            action="Use generated built-in reference directions; planning never reads an external source path.",
        )
    path = raw.get("path")
    divisions = raw.get("divisions")
    if path is not None:
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="UNSAFE_REFERENCE_DIRECTIONS",
            field=f"$.matrix[{candidate.index}].algorithm_config.reference_directions.path",
            expected="null (built-in generated reference directions)",
            actual="external path",
            action="Remove the reference-direction path; planning never follows external scientific inputs.",
        )
    if isinstance(divisions, bool) or not isinstance(divisions, int) or divisions < 1:
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="INVALID_REFERENCE_DIRECTIONS",
            field=f"$.matrix[{candidate.index}].algorithm_config.reference_directions.divisions",
            expected="positive integer",
            actual=divisions,
            action="Choose a positive reference-direction division count.",
        )
    expected = comb(divisions + n_obj - 1, n_obj - 1)
    if population < expected:
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="REFERENCE_DIRECTION_POPULATION_MISMATCH",
            field=f"$.matrix[{candidate.index}].pop_size",
            expected=f">= {expected} for n_obj={n_obj} and divisions={divisions}",
            actual=population,
            action="Use at least the reference-direction count as pop_size or choose fewer divisions.",
        )


def _validate_rvea_reference_directions(config: Mapping[str, Any], *, population: int, n_obj: int, candidate: _Candidate) -> None:
    partitions = config.get("n_partitions")
    if isinstance(partitions, bool) or not isinstance(partitions, int) or partitions < 1:
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="INVALID_REFERENCE_DIRECTIONS",
            field=f"$.matrix[{candidate.index}].algorithm_config.n_partitions",
            expected="positive integer",
            actual=partitions,
            action="Choose a positive RVEA reference-vector partition count.",
        )
    expected = comb(partitions + n_obj - 1, n_obj - 1)
    if population != expected:
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="REFERENCE_DIRECTION_POPULATION_MISMATCH",
            field=f"$.matrix[{candidate.index}].pop_size",
            expected=f"{expected} for n_obj={n_obj} and n_partitions={partitions}",
            actual=population,
            action="Use the reference-vector count as pop_size or choose compatible n_partitions.",
        )


def _validate_resolved_components(resolved: Mapping[str, Any], *, candidate: _Candidate) -> None:
    component_fields: list[tuple[str, object]] = [
        ("problem", resolved.get("problem")),
        ("algorithm", resolved.get("algorithm")),
        ("termination", resolved.get("termination")),
    ]
    operators = resolved.get("operators")
    if isinstance(operators, Mapping):
        component_fields.extend((f"operators.{name}", value) for name, value in operators.items())
    backends = resolved.get("backend")
    if isinstance(backends, Mapping):
        component_fields.extend((f"backend.{name}", value) for name, value in backends.items())
    for field, raw in component_fields:
        if not isinstance(raw, Mapping):
            raise AssertionError(f"canonical run resolver omitted {field}")
        provider = raw.get("provider")
        if not isinstance(provider, Mapping) or provider.get("type") != "built_in":
            raise UnresolvedStudyTaskError(
                operation="resolve study plan",
                reason="UNAVAILABLE_COMPONENT",
                field=f"$.matrix[{candidate.index}].{field}",
                expected="supported built-in component",
                actual=raw.get("component_id"),
                action="Choose a supported built-in component; planning never discovers or imports plugins.",
            )
    if isinstance(backends, Mapping):
        _validate_resolved_backends(backends, candidate=candidate)


def _validate_resolved_backends(backends: Mapping[str, object], *, candidate: _Candidate) -> None:
    kernel = backends.get("kernel")
    if isinstance(kernel, Mapping):
        resolution = kernel.get("resolution")
        if not isinstance(resolution, Mapping) or resolution.get("version") is None:
            raise UnresolvedStudyTaskError(
                operation="resolve study plan",
                reason="BACKEND_UNAVAILABLE",
                field=f"$.matrix[{candidate.index}].backend.kernel",
                expected="installed supported kernel backend",
                actual=kernel.get("component_id"),
                action="Install the requested backend dependency or choose an available built-in backend.",
            )
    evaluation = backends.get("evaluation")
    if isinstance(evaluation, Mapping) and not is_reconstructable_evaluation_backend(evaluation):
        raise UnresolvedStudyTaskError(
            operation="resolve study plan",
            reason="UNSUPPORTED_EVALUATION_BACKEND",
            field=f"$.matrix[{candidate.index}].backend.evaluation",
            expected="serial evaluation, the only backend Study.run() executes",
            actual=evaluation.get("component_id"),
            action="Use eval_strategy='serial' in the StudySpec; run multiprocessing or Dask evaluation with vamos.optimize().",
        )


def _reject_duplicates(tasks: tuple[PlanTask, ...]) -> None:
    seen: set[str] = set()
    for task in tasks:
        if task.task_id in seen:
            raise DuplicateStudyTaskError(
                operation="resolve study plan",
                reason="DUPLICATE_CANONICAL_TASK",
                entity_id=task.task_id,
                expected="one matrix candidate per scientific task",
                actual="duplicate resolved run specification",
                action="Remove duplicate matrix values; no study directory was published.",
            )
        seen.add(task.task_id)


def _plan_hash(plan: ResolvedStudyPlan) -> str:
    tasks = [
        {
            "plan_index": task.plan_index,
            "requested_run": deep_thaw(task.requested_run),
            "resolved_run_spec": deep_thaw(task.resolved_run_spec),
            "task_id": task.task_id,
            "task_spec_sha256": task.task_spec_sha256,
        }
        for task in plan.tasks
    ]
    return document_self_hash(
        {
            "document_type": "vamos.resolved-study-plan",
            "schema_version": SCHEMA_VERSION,
            "plan_id": plan.plan_id,
            "task_count": len(tasks),
            "tasks": tasks,
            "integrity": {},
        }
    )


__all__ = ["resolve_spec"]
