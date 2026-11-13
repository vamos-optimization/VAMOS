"""Data-only built-in component reconstructability assessment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vamos.foundation.problem.registry import get_problem_specs

from .models import RunManifest
from .reports import ComponentStatus, VerificationReason

BUILT_IN_ALGORITHMS = ("agemoea", "ibea", "moead", "nsgaii", "nsgaiii", "rvea", "smpso", "smsemoa", "spea2")
BUILT_IN_BACKENDS = ("numpy", "numba", "moocore")
BUILT_IN_OPERATORS = (
    "bitflip",
    "clip",
    "differential_evolution",
    "mixed",
    "neighborhood",
    "order",
    "polynomial",
    "random",
    "sbx",
    "swap",
    "tournament",
    "uniform",
)
BUILT_IN_TERMINATIONS = ("max_evaluations", "n_gen", "hv")


def component_reconstructability(manifest: RunManifest) -> tuple[ComponentStatus, tuple[VerificationReason, ...]]:
    """Assess identifiers/providers without resolving or instantiating components."""
    return resolved_component_reconstructability(manifest.resolved_spec)


def resolved_component_reconstructability(
    resolved: Mapping[str, Any],
) -> tuple[ComponentStatus, tuple[VerificationReason, ...]]:
    """Assess a persisted resolved spec without requiring a RunManifest."""
    checks = (
        _check_descriptor(resolved.get("problem"), "problem", "vamos.problem", frozenset(get_problem_specs())),
        _check_descriptor(resolved.get("algorithm"), "algorithm", "vamos.algorithm", BUILT_IN_ALGORITHMS),
        _check_backend(resolved.get("backend")),
        _check_descriptor(resolved.get("termination"), "termination", "vamos.termination", BUILT_IN_TERMINATIONS),
        _check_operators(resolved.get("operators")),
    )
    reasons = tuple(reason for group in checks for reason in group)
    unavailable_field = _unavailable_field(resolved)
    if unavailable_field is not None:
        reasons = (
            *reasons,
            _reason(
                "unavailable_configuration", unavailable_field, "Executable or unavailable configuration data cannot be reconstructed."
            ),
        )
    if not reasons:
        return "reconstructable", ()
    status: ComponentStatus = "manual" if any(reason.code == "custom_provider" for reason in reasons) else "unavailable"
    return status, reasons


def _check_descriptor(
    value: object, field: str, namespace: str, supported: tuple[str, ...] | frozenset[str]
) -> tuple[VerificationReason, ...]:
    if not isinstance(value, Mapping):
        return (_reason("missing_component", f"$.resolved_spec.{field}", "The component descriptor is missing."),)
    provider_reason = _provider_reason(value, field)
    if provider_reason is not None:
        return (provider_reason,)
    name = _stable_component_name(value.get("component_id"), namespace)
    if name is None or name not in supported:
        return (
            _reason(
                "unsupported_component",
                f"$.resolved_spec.{field}.component_id",
                f"The built-in component ID is not supported: {value.get('component_id')!r}.",
            ),
        )
    return ()


def _check_backend(value: object) -> tuple[VerificationReason, ...]:
    if not isinstance(value, Mapping):
        return (_reason("missing_component", "$.resolved_spec.backend", "The backend descriptor is missing."),)
    kernel = _check_descriptor(value.get("kernel"), "backend.kernel", "vamos.kernel", BUILT_IN_BACKENDS)
    evaluation = _check_descriptor(value.get("evaluation"), "backend.evaluation", "vamos.evaluation", frozenset({"serial"}))
    return (*kernel, *evaluation)


def _check_operators(value: object) -> tuple[VerificationReason, ...]:
    if not isinstance(value, Mapping):
        return (_reason("missing_component", "$.resolved_spec.operators", "The operator map is missing."),)
    reasons: list[VerificationReason] = []
    for role, descriptor in value.items():
        field = f"operators.{role}"
        if not isinstance(descriptor, Mapping):
            reasons.append(_reason("missing_component", f"$.resolved_spec.{field}", "The operator descriptor is malformed."))
            continue
        resolution = descriptor.get("resolution")
        if isinstance(resolution, Mapping) and resolution.get("active") is False:
            continue
        reasons.extend(_check_descriptor(descriptor, field, "vamos.operator", BUILT_IN_OPERATORS))
    return tuple(reasons)


def _provider_reason(value: Mapping[str, Any], field: str) -> VerificationReason | None:
    provider = value.get("provider")
    provider_type = provider.get("type") if isinstance(provider, Mapping) else None
    distribution = provider.get("distribution") if isinstance(provider, Mapping) else None
    if provider_type == "built_in" and distribution == "vamos-optimization":
        return None
    code = "custom_provider" if provider_type == "custom_python" else "unsupported_provider"
    return _reason(
        code,
        f"$.resolved_spec.{field}.provider",
        f"Provider {provider_type!r} is outside the built-in replay trust boundary.",
    )


def _stable_component_name(value: object, namespace: str) -> str | None:
    prefix = f"{namespace}:"
    if not isinstance(value, str) or not value.startswith(prefix) or not value.endswith("@1"):
        return None
    name = value[len(prefix) : -2]
    if not name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for character in name):
        return None
    return name


def _reason(code: str, field: str, message: str) -> VerificationReason:
    return VerificationReason(
        code=code,
        field=field,
        message=message,
        action="Regenerate the run using complete registered VAMOS built-in components.",
    )


def _unavailable_field(value: object, field: str = "$.resolved_spec") -> str | None:
    if isinstance(value, Mapping):
        if "unavailable_reason" in value:
            return field
        for key, item in value.items():
            found = _unavailable_field(item, f"{field}.{key}")
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _unavailable_field(item, f"{field}[{index}]")
            if found is not None:
                return found
    return None


__all__ = [
    "BUILT_IN_ALGORITHMS",
    "BUILT_IN_BACKENDS",
    "BUILT_IN_OPERATORS",
    "BUILT_IN_TERMINATIONS",
    "component_reconstructability",
    "resolved_component_reconstructability",
]
