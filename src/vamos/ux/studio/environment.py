"""What this installation can run: the VAMOS version, kernel backends and optional extras.

Everything is read from installed-package metadata; no backend is imported.
"""

from __future__ import annotations

import platform
import re
from dataclasses import dataclass
from importlib import metadata

DISTRIBUTION = "vamos-optimization"
KERNEL_EXTRAS: dict[str, str | None] = {"numpy": None, "numba": "compute", "moocore": "compute"}
# Optional extras that change what Studio and VAMOS can do, with what each adds.
EXTRA_PURPOSES: dict[str, str] = {
    "compute": "Numba and MooCore kernels, Dask evaluation",
    "analysis": "plots and notebook analysis",
    "tuning": "Optuna and SMAC tuning backends",
    "research": "comparisons with other frameworks",
    "openai": "OpenAI code generation in Problem Builder",
    "anthropic": "Anthropic code generation in Problem Builder",
}
_NAME = r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)"
_EXTRA_MARKER = r"""^extra\s*==\s*["']([^"']+)["']$"""


@dataclass(frozen=True)
class Capability:
    """One backend or extra: whether it is installed, and its version or how to install it."""

    name: str
    available: bool
    detail: str


@dataclass(frozen=True)
class EnvironmentReport:
    vamos_version: str | None
    python_version: str
    kernels: tuple[Capability, ...]
    extras: tuple[Capability, ...]


def environment_report() -> EnvironmentReport:
    """Describe the running installation from package metadata."""
    vamos_version = installed_version(DISTRIBUTION)
    requirements = _extra_requirements() if vamos_version is not None else {}
    kernels = tuple(_kernel(name, extra) for name, extra in KERNEL_EXTRAS.items())
    extras = tuple(_extra(name, requirements.get(name)) for name in EXTRA_PURPOSES)
    return EnvironmentReport(
        vamos_version=vamos_version,
        python_version=platform.python_version(),
        kernels=kernels,
        extras=extras,
    )


def environment_markdown(report: EnvironmentReport) -> str:
    """The report as compact Markdown for the Studio sidebar."""
    version = report.vamos_version or "unknown (not installed as a package)"
    lines = [f"**VAMOS** {version} · Python {report.python_version}", "", "**Kernels**"]
    lines += [f"- {_mark(item)} {item.name}: {item.detail}" for item in report.kernels]
    lines += ["", "**Extras**"]
    lines += [f"- {_mark(item)} `{item.name}`: {item.detail}" for item in report.extras]
    return "\n".join(lines)


def _kernel(name: str, extra: str | None) -> Capability:
    version = installed_version(name)
    if version is not None:
        return Capability(name=name, available=True, detail=version)
    return Capability(name=name, available=False, detail=f"not installed (extra `{extra}`)" if extra else "not installed")


def _extra(name: str, requirements: tuple[str, ...] | None) -> Capability:
    purpose = EXTRA_PURPOSES[name]
    if requirements is None:
        return Capability(name=name, available=False, detail=f"{purpose} — unknown")
    missing = [requirement for requirement in requirements if installed_version(requirement) is None]
    if not missing:
        return Capability(name=name, available=True, detail=purpose)
    return Capability(name=name, available=False, detail=f"{purpose} — missing {', '.join(missing)}")


def _extra_requirements() -> dict[str, tuple[str, ...]]:
    """Distribution names required by each extra; conditional requirements are left out."""
    found: dict[str, list[str]] = {}
    for line in metadata.requires(DISTRIBUTION) or []:
        requirement, _, marker = line.partition(";")
        extra = re.match(_EXTRA_MARKER, marker.strip())
        name = re.match(_NAME, requirement)
        if extra is not None and name is not None:
            found.setdefault(extra.group(1), []).append(name.group(1))
    return {extra: tuple(names) for extra, names in found.items()}


def installed_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _mark(item: Capability) -> str:
    return "✓" if item.available else "✗"


__all__ = ["Capability", "EnvironmentReport", "environment_markdown", "environment_report", "installed_version"]
