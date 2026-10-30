"""Pure current-environment comparison for canonical run verification."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .models import RunManifest, deep_freeze
from .provenance import capture_runtime_evidence
from .reports import CompatibilityFinding, CompatibilityLevel, CompatibilityReport


def compare_current_environment(manifest: RunManifest, stored_environment: Mapping[str, Any]) -> CompatibilityReport:
    """Compare scientifically material evidence without shell, network, or mutation."""
    backend = _backend_name(manifest.resolved_spec)
    current_evidence, current_environment = capture_runtime_evidence(backend=backend)
    pairs = _material_pairs(manifest, stored_environment, current_evidence, current_environment)
    findings = tuple(_finding(field, stored, current, explanation) for field, stored, current, explanation in pairs)
    return CompatibilityReport(level=_overall_level(findings), findings=findings)


def _material_pairs(
    manifest: RunManifest,
    stored_environment: Mapping[str, Any],
    current_evidence: Mapping[str, Any],
    current_environment: Mapping[str, Any],
) -> tuple[tuple[str, Any, Any, str], ...]:
    provenance = manifest.get("provenance")
    stored_provenance = provenance if isinstance(provenance, Mapping) else {}
    stored_impl = _mapping(stored_provenance.get("implementation"))
    stored_distribution = _mapping(stored_impl.get("distribution"))
    stored_source = _mapping(stored_provenance.get("source"))
    current_impl = _mapping(current_evidence.get("implementation"))
    current_distribution = _mapping(current_impl.get("distribution"))
    current_source = _mapping(current_evidence.get("source"))
    stored_python = _mapping(stored_environment.get("python"))
    current_python = _mapping(current_environment.get("python"))
    stored_platform = _mapping(stored_environment.get("platform"))
    current_platform = _mapping(current_environment.get("platform"))
    stored_packages = _mapping(stored_environment.get("packages"))
    current_packages = _mapping(current_environment.get("packages"))
    stored_backend = _mapping(stored_environment.get("backend"))
    current_backend = _mapping(current_environment.get("backend"))
    return (
        (
            "$.environment.document_type",
            stored_environment.get("document_type"),
            current_environment.get("document_type"),
            "environment document identity",
        ),
        (
            "$.environment.schema_version",
            stored_environment.get("schema_version"),
            current_environment.get("schema_version"),
            "environment schema",
        ),
        ("$.provenance.implementation.vamos_version", stored_impl.get("vamos_version"), current_impl.get("vamos_version"), "VAMOS version"),
        (
            "$.provenance.implementation.distribution.sha256",
            stored_distribution.get("sha256"),
            current_distribution.get("sha256"),
            "VAMOS implementation fingerprint",
        ),
        ("$.provenance.source.kind", stored_source.get("kind"), current_source.get("kind"), "installed source kind"),
        (
            "$.provenance.source.git_sha",
            _checkout_value(stored_source, "git_sha"),
            _checkout_value(current_source, "git_sha"),
            "checkout Git revision",
        ),
        (
            "$.provenance.source.dirty",
            _checkout_value(stored_source, "dirty"),
            _current_dirty_state(stored_source, current_source, stored_distribution, current_distribution),
            "checkout scientific source state confirmed by the implementation fingerprint",
        ),
        (
            "$.provenance.source.package_data_sha256",
            _package_data_value(stored_source, stored_source),
            _package_data_value(stored_source, current_source),
            "checkout packaged-data fingerprint",
        ),
        (
            "$.environment.python.implementation",
            stored_python.get("implementation"),
            current_python.get("implementation"),
            "Python implementation",
        ),
        (
            "$.environment.python.version",
            _major_minor(stored_python.get("version")),
            _major_minor(current_python.get("version")),
            "Python major/minor version",
        ),
        (
            "$.environment.platform.operating_system",
            stored_platform.get("operating_system"),
            current_platform.get("operating_system"),
            "operating system",
        ),
        (
            "$.environment.platform.architecture",
            stored_platform.get("architecture"),
            current_platform.get("architecture"),
            "machine architecture",
        ),
        ("$.environment.packages.numpy", stored_packages.get("numpy"), current_packages.get("numpy"), "NumPy version"),
        ("$.environment.packages.scipy", stored_packages.get("scipy"), current_packages.get("scipy"), "SciPy version"),
        ("$.environment.backend.name", stored_backend.get("name"), current_backend.get("name"), "selected backend"),
        (
            "$.environment.backend.package",
            stored_backend.get("package"),
            current_backend.get("package"),
            "backend package identity/version",
        ),
        ("$.environment.blas", stored_environment.get("blas"), current_environment.get("blas"), "BLAS vendor and integer width"),
        ("$.environment.threads", stored_environment.get("threads"), current_environment.get("threads"), "material thread controls"),
        (
            "$.resolved_spec.backend.kernel.resolution.capabilities",
            _backend_capabilities(manifest.resolved_spec),
            (),
            "persisted backend capabilities supported by this contract",
        ),
    )


def _finding(field: str, stored: Any, current: Any, explanation: str) -> CompatibilityFinding:
    if stored is None or current is None:
        classification: CompatibilityLevel = "unavailable"
        message = f"Required {explanation} evidence is missing."
        action = "Regenerate the source run in an environment that records complete exact-replay evidence."
    elif stored == current:
        classification = "exact"
        message = f"Stored and current {explanation} match."
        action = "No action required."
    else:
        classification = "compatible"
        message = f"Stored and current {explanation} differ."
        action = "Use the same runtime and dependency build that produced the source run."
    return CompatibilityFinding(
        field=field,
        stored=deep_freeze(stored),
        current=deep_freeze(current),
        classification=classification,
        explanation=message,
        blocks_exact=classification != "exact",
        action=action,
    )


def _overall_level(findings: tuple[CompatibilityFinding, ...]) -> CompatibilityLevel:
    levels = {item.classification for item in findings if item.blocks_exact}
    if "unavailable" in levels:
        return "unavailable"
    if levels:
        return "compatible"
    return "exact"


def _backend_name(resolved: Mapping[str, Any]) -> str:
    backend = _mapping(resolved.get("backend"))
    kernel = _mapping(backend.get("kernel"))
    resolution = _mapping(kernel.get("resolution"))
    name = resolution.get("name")
    if isinstance(name, str):
        return name
    component_id = kernel.get("component_id")
    return _component_name(component_id) or "unavailable"


def _backend_capabilities(resolved: Mapping[str, Any]) -> Any:
    backend = _mapping(resolved.get("backend"))
    kernel = _mapping(backend.get("kernel"))
    return _mapping(kernel.get("resolution")).get("capabilities")


def _component_name(value: object) -> str | None:
    if not isinstance(value, str) or ":" not in value or "@" not in value:
        return None
    return value.split(":", 1)[1].split("@", 1)[0]


def _git_checkout(source: Mapping[str, Any]) -> bool:
    """A checkout whose sources Git versions; an editable install without Git has no revision or dirty state."""
    return source.get("kind") == "checkout" and source.get("reason") != "git_checkout_unavailable"


def _checkout_value(source: Mapping[str, Any], key: str) -> object:
    return source.get(key) if _git_checkout(source) else "not_applicable"


def _package_data_value(stored_source: Mapping[str, Any], source: Mapping[str, Any]) -> object:
    # Runs saved before this fingerprint existed, and wheels (whose distribution hash covers their data), skip it.
    if "package_data_sha256" not in stored_source:
        return "not_applicable"
    return source.get("package_data_sha256", "not_applicable") if source.get("kind") == "checkout" else "not_applicable"


def _current_dirty_state(
    stored_source: Mapping[str, Any],
    current_source: Mapping[str, Any],
    stored_distribution: Mapping[str, Any],
    current_distribution: Mapping[str, Any],
) -> object:
    if not (_git_checkout(stored_source) and _git_checkout(current_source)):
        return "not_applicable"
    fingerprint_matches = stored_distribution.get("sha256") is not None and stored_distribution.get("sha256") == current_distribution.get(
        "sha256"
    )
    revision_matches = stored_source.get("git_sha") is not None and stored_source.get("git_sha") == current_source.get("git_sha")
    if fingerprint_matches and revision_matches:
        return stored_source.get("dirty")
    return "scientific_content_changed"


def _major_minor(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    parts = value.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else None


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


__all__ = ["compare_current_environment"]
