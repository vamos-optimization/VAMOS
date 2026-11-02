"""Executable exact replay for verified canonical built-in runs."""

from __future__ import annotations

import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vamos.experiment.optimize import _OptimizeConfig, _run_config

from .bundle import snapshot_result_arrays
from .comparison import compare_array_collections, comparisons_are_exact
from .errors import EnvironmentIncompatibilityError, ReplayExecutionError, ReplayResultMismatchError, sanitized_message
from .models import LoadLimits, deep_thaw
from .persistence import load_run, save_failed_replay, save_result
from .reconstruction import ReconstructedRun, ReplayPlan, build_replay_plan, instantiate_reconstructed_problem
from .reports import ArrayComparison, ReplayReport
from .verification import verify_run


def reproduce(
    path: str | Path,
    *,
    output: str | Path | None = None,
    limits: LoadLimits | None = None,
) -> ReplayReport:
    """Verify, exactly execute, compare, and store a new built-in replay."""
    active_limits = limits if limits is not None else LoadLimits()
    verification = verify_run(path, limits=active_limits)
    if verification.environment.level != "exact":
        raise EnvironmentIncompatibilityError(
            operation="reproduce run",
            field="$.environment",
            path=verification.root,
            reason="does not satisfy exact replay compatibility",
            expected="exact",
            actual=verification.environment.level,
            action="Use the same implementation, Python, dependencies, backend, BLAS, and material thread settings.",
            optimization_executed=False,
        )
    source = load_run(path, verify="all", limits=active_limits)
    plan = build_replay_plan(source, verification, output)
    source_arrays = snapshot_result_arrays(source.result, limits=active_limits)
    problem = _instantiate_problem(plan)
    started_at = _now()
    started_monotonic = time.perf_counter()
    try:
        result = _run_config(
            _OptimizeConfig(
                problem=problem,
                algorithm=plan.algorithm,
                algorithm_config=plan.algorithm_config,
                termination=plan.termination,
                seed=plan.seed,
                engine=plan.engine,
                eval_strategy=plan.eval_strategy,
            ),
            built_in_only=True,
        )
    except Exception as exc:
        completed_at = _now()
        runtime_ms = (time.perf_counter() - started_monotonic) * 1000.0
        _store_failed_attempt(plan, exc, started_at, completed_at, runtime_ms, active_limits)
        raise ReplayExecutionError(
            operation="reproduce run",
            field="$.lineage.comparison",
            path=plan.output_root,
            reason="optimization execution failed",
            expected="completed deterministic built-in execution",
            actual={"exception_type": type(exc).__name__, "message": _sanitized_message(exc)},
            action=f"Inspect the failed canonical attempt at {plan.output_root} and correct the built-in execution failure.",
            optimization_executed=True,
        ) from exc
    completed_at = _now()
    runtime_ms = (time.perf_counter() - started_monotonic) * 1000.0
    replay_arrays = snapshot_result_arrays(result, limits=active_limits)
    comparisons = compare_array_collections(source_arrays, replay_arrays)
    exact = comparisons_are_exact(comparisons)
    lineage = _lineage(plan, comparisons, status="exact_match" if exact else "mismatch")
    _attach_replay_metadata(result, plan, lineage, started_at, completed_at, runtime_ms, exact)
    stored = save_result(result, plan.output_root, limits=active_limits)
    report = ReplayReport(
        source_root=source.root,
        output_root=stored.root,
        source_run_id=plan.source_run_id,
        replay_run_id=stored.manifest.run_id,
        task_id=stored.manifest.task_id,
        source_manifest_sha256=plan.source_manifest_sha256,
        replay_plan_sha256=plan.replay_plan_sha256,
        exact=exact,
        comparisons=comparisons,
        verification=verification,
    )
    if not exact:
        raise ReplayResultMismatchError(
            operation="reproduce run",
            field="$.lineage.comparison",
            path=stored.root,
            reason="completed without bitwise equality",
            expected="exact F, X, and auxiliary deterministic arrays",
            actual={"exact": False, "comparisons": [item.as_dict() for item in comparisons if not item.exact]},
            action=f"Inspect the stored mismatch attempt at {stored.root}; do not treat it as an exact replay.",
            optimization_executed=True,
        )
    return report


def _instantiate_problem(plan: ReplayPlan) -> Any:
    reconstructed = ReconstructedRun(
        resolved_spec=plan.resolved_spec,
        problem=plan.problem,
        n_var=plan.n_var,
        n_obj=plan.n_obj,
        encoding=plan.encoding,
        algorithm=plan.algorithm,
        algorithm_config=plan.algorithm_config,
        termination=plan.termination,
        engine=plan.engine,
        eval_strategy=plan.eval_strategy,
        seed=plan.seed,
    )
    return instantiate_reconstructed_problem(reconstructed, root=plan.source_root)


def _attach_replay_metadata(
    result: Any,
    plan: ReplayPlan,
    lineage: Mapping[str, Any],
    started_at: str,
    completed_at: str,
    runtime_ms: float,
    exact: bool,
) -> None:
    result.meta["run_artifact_requested_spec"] = deep_thaw(plan.requested_spec)
    result.meta["run_artifact_resolved_spec"] = deep_thaw(plan.resolved_spec)
    result.meta["run_artifact_timestamps"] = {
        "started_at": started_at,
        "completed_at": completed_at,
        "runtime_ms": runtime_ms,
    }
    result.meta["run_artifact_entry_point"] = {
        "kind": "replay",
        "python": {"callable": "vamos.reproduce", "arguments_source": "resolved_spec"},
    }
    result.meta["run_artifact_run_id"] = plan.new_run_id
    result.meta["run_artifact_lineage"] = dict(lineage)
    result.meta["run_artifact_replayability"] = {
        "declared_level": "exact" if exact else "unavailable",
        "deterministic": True,
        "exact_requirements": [
            "same_resolved_spec",
            "same_implementation",
            "same_backend",
            "materially_equivalent_environment",
            "bitwise_equal_result",
        ]
        if exact
        else [],
        "reasons": []
        if exact
        else [{"code": "exact_comparison_mismatch", "message": "At least one deterministic result array differed bitwise."}],
    }


def _store_failed_attempt(
    plan: ReplayPlan,
    exc: Exception,
    started_at: str,
    completed_at: str,
    runtime_ms: float,
    limits: LoadLimits,
) -> None:
    lineage = _lineage(plan, (), status="execution_failed")
    save_failed_replay(
        plan.output_root,
        run_id=plan.new_run_id,
        requested_spec=plan.requested_spec,
        resolved_spec=plan.resolved_spec,
        lineage=lineage,
        failure={
            "phase": "optimization",
            "exception_type": type(exc).__name__,
            "message": _sanitized_message(exc),
            "traceback": None,
            "optimization_executed": True,
        },
        outcome={
            "evaluations": None,
            "generations": None,
            "runtime_ms": runtime_ms,
            "termination_reason": "replay_execution_error",
            "result_mode": _result_mode(plan),
            "interrupted": True,
            "usable_result": False,
            "n_solutions": None,
            "n_objectives": plan.n_obj,
            "n_variables": plan.n_var,
            "metrics": {},
        },
        timestamps={"started_at": started_at, "completed_at": completed_at},
        limits=limits,
    )


def _lineage(plan: ReplayPlan, comparisons: tuple[ArrayComparison, ...], *, status: str) -> dict[str, Any]:
    return {
        "execution_kind": "replay",
        "source_run_id": plan.source_run_id,
        "source_manifest_sha256": plan.source_manifest_sha256,
        "root_run_id": plan.root_run_id,
        "depth": plan.lineage_depth,
        "replay_plan_sha256": plan.replay_plan_sha256,
        "compatibility_level": "exact",
        "comparison": {
            "status": status,
            "arrays": [item.as_dict() for item in comparisons],
        },
    }


def _result_mode(plan: ReplayPlan) -> str:
    value = plan.algorithm_config.to_dict().get("result_mode")
    return str(value) if value is not None else "unspecified"


def _sanitized_message(exc: Exception) -> str:
    return sanitized_message(exc, empty="Built-in replay execution failed without a message.")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = ["reproduce"]
