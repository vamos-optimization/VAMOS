"""Full inert verification of canonical run artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

from .bundle import inspect_result_bundle
from .compatibility import compare_current_environment
from .component_support import component_reconstructability
from .errors import EnvironmentIncompatibilityError, ReplayUnavailableError, VerificationRequirementError
from .models import LoadLimits, StoredRun
from .paths import confined_artifact_path
from .reader import read_run, verify_artifact
from .reports import ReplayabilityLevel, VerificationReason, VerificationReport
from .resolved_reconstruction import reconstruct_resolved_run

RequiredLevel = Literal["exact", "compatible", "best_effort", "manual", "unavailable"]
_LEVEL_RANK = {"unavailable": 0, "manual": 1, "best_effort": 2, "compatible": 3, "exact": 4}


def verify_run(
    path: str | Path,
    *,
    require_level: RequiredLevel | None = None,
    limits: LoadLimits | None = None,
) -> VerificationReport:
    """Fully verify a canonical run without executing or resolving code."""
    active_limits = limits if limits is not None else LoadLimits()
    stored = read_run(path, verify="all", limits=active_limits)
    _verify_every_artifact(stored, active_limits)
    bundle_status = _inspect_numerical_bundle(stored, active_limits)
    environment = compare_current_environment(stored.manifest, stored.environment)
    component_status, component_reasons = component_reconstructability(stored.manifest)
    if component_status == "reconstructable":
        component_status, component_reasons = _reconstruction_status(stored, component_reasons)
    reasons = (*component_reasons, *_effective_reasons(stored, environment.level, bundle_status))
    effective = _effective_level(stored, environment.level, component_status, bundle_status)
    report = VerificationReport(
        root=stored.root,
        run_id=stored.manifest.run_id,
        task_id=stored.manifest.task_id,
        status=stored.status,
        schema=str(stored.manifest["schema_version"]),
        artifact_integrity="valid",
        path_safety="valid",
        numerical_bundle_safety=bundle_status,
        environment=environment,
        component_reconstructability=component_status,
        effective_replayability=effective,
        reasons=reasons,
    )
    _enforce_requirement(report, require_level)
    return report


def _reconstruction_status(
    stored: StoredRun, reasons: tuple[VerificationReason, ...]
) -> tuple[Literal["reconstructable", "unavailable"], tuple[VerificationReason, ...]]:
    """Prove the resolved spec with the reconstruction ``reproduce`` uses, without running the optimization."""
    try:
        reconstruct_resolved_run(stored.manifest.resolved_spec, root=stored.root)
    except ReplayUnavailableError as exc:
        field = exc.field or "$.resolved_spec"
        return "unavailable", (*reasons, _reason("component_not_reconstructable", field, str(exc)))
    return "reconstructable", reasons


def _verify_every_artifact(stored: StoredRun, limits: LoadLimits) -> None:
    for descriptor in stored.manifest.artifacts:
        verify_artifact(stored.root, descriptor, limits=limits, operation="verify run")


def _inspect_numerical_bundle(stored: StoredRun, limits: LoadLimits) -> Literal["valid", "not_fully_checked"]:
    descriptor = stored.manifest.artifact("result_bundle")
    if descriptor is None:
        return "not_fully_checked"
    path = confined_artifact_path(stored.root, descriptor.path, role=descriptor.role, operation="verify run", must_exist=True)
    inspect_result_bundle(
        path,
        descriptor=descriptor,
        limits=limits,
        required_f=stored.status == "succeeded",
        operation="verify run",
    )
    return "valid"


def _effective_level(
    stored: StoredRun,
    environment_level: str,
    component_status: str,
    bundle_status: str,
) -> ReplayabilityLevel:
    replayability = stored.manifest.get("replayability")
    stored_level = replayability.get("declared_level") if isinstance(replayability, Mapping) else "unavailable"
    candidates = [str(stored_level), environment_level]
    candidates.append({"reconstructable": "exact", "manual": "manual", "unavailable": "unavailable"}[component_status])
    if stored.status != "succeeded" or bundle_status != "valid" or not _has_mandatory_arrays(stored):
        candidates.append("unavailable")
    determinism = stored.manifest.resolved_spec.get("determinism")
    if not isinstance(determinism, Mapping) or determinism.get("declared") is not True:
        candidates.append("best_effort")
    return cast(ReplayabilityLevel, min(candidates, key=lambda value: _LEVEL_RANK.get(value, 0)))


def _effective_reasons(stored: StoredRun, environment_level: str, bundle_status: str) -> tuple[VerificationReason, ...]:
    reasons: list[VerificationReason] = []
    if environment_level != "exact":
        reasons.append(_reason("environment_not_exact", "$.environment", "The current environment is not an exact match."))
    if stored.status != "succeeded":
        reasons.append(_reason("run_not_succeeded", "$.status", f"Run status {stored.status!r} cannot be replayed exactly."))
    if bundle_status != "valid":
        reasons.append(_reason("result_bundle_unavailable", "$.artifacts", "No verified numerical result bundle is available."))
    if not _has_mandatory_arrays(stored):
        reasons.append(
            _reason("mandatory_arrays_missing", "$.artifacts[result_bundle].array_contract", "Exact replay requires stored F and X arrays.")
        )
    replayability = stored.manifest.get("replayability")
    declared = replayability.get("declared_level") if isinstance(replayability, Mapping) else None
    if declared != "exact":
        reasons.append(_reason("stored_level_not_exact", "$.replayability.declared_level", f"Stored replayability is {declared!r}."))
    return tuple(reasons)


def _has_mandatory_arrays(stored: StoredRun) -> bool:
    descriptor = stored.manifest.artifact("result_bundle")
    contract = descriptor.array_contract if descriptor is not None else None
    return isinstance(contract, Mapping) and {"F", "X"}.issubset(contract)


def _enforce_requirement(report: VerificationReport, required: RequiredLevel | None) -> None:
    if required is None:
        return
    if required not in _LEVEL_RANK:
        raise ValueError("require_level must be 'exact', 'compatible', 'best_effort', 'manual', 'unavailable', or None.")
    if _LEVEL_RANK[report.effective_replayability] >= _LEVEL_RANK[required]:
        return
    error_type = EnvironmentIncompatibilityError if report.environment.level != "exact" else VerificationRequirementError
    raise error_type(
        operation="verify run",
        field="$.effective_replayability",
        path=report.root,
        reason="does not meet the required replayability level",
        expected=required,
        actual=report.effective_replayability,
        action="Use the recorded exact environment and supported built-in components, then verify again.",
        optimization_executed=False,
    )


def _reason(code: str, field: str, message: str) -> VerificationReason:
    return VerificationReason(
        code=code,
        field=field,
        message=message,
        action="Inspect the structured verification findings before attempting replay.",
    )


__all__ = ["RequiredLevel", "verify_run"]
