"""The dependencies that pyproject.toml declares, by distribution name and by import name."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"

# Declared distributions whose import name is not the distribution name with "-" as "_".
# Without an entry, the policy tests would watch a module name that is never imported.
IMPORT_NAMES: dict[str, tuple[str, ...]] = {
    "configspace": ("ConfigSpace",),
    "google-genai": ("google.genai",),
    "ipython": ("IPython",),
    "jmetalpy": ("jmetal",),
    "mkdocs-material": ("material",),
    "platypus-opt": ("platypus",),
    "pymdown-extensions": ("pymdownx",),
    "pyyaml": ("yaml",),
    "scikit-learn": ("sklearn",),
}

_REQUIREMENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def distribution_name(requirement: str) -> str:
    """Return the normalized (PEP 503) distribution name of a PEP 508 requirement."""
    match = _REQUIREMENT_NAME.match(requirement.strip())
    if match is None:
        raise ValueError(f"not a requirement: {requirement!r}")
    return re.sub(r"[-_.]+", "-", match.group(0)).lower()


def import_names(distribution: str) -> tuple[str, ...]:
    """Return the dotted module names that a declared distribution provides."""
    return IMPORT_NAMES.get(distribution, (distribution.replace("-", "_"),))


def _project() -> dict[str, object]:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    assert isinstance(project, dict)
    return project


def base_distributions() -> set[str]:
    """Return the distributions of ``[project].dependencies``."""
    requirements = _project()["dependencies"]
    assert isinstance(requirements, list)
    return {distribution_name(requirement) for requirement in requirements}


def optional_distributions() -> dict[str, set[str]]:
    """Return the distributions of each extra that are not base dependencies, without the ``all`` union."""
    extras = _project()["optional-dependencies"]
    assert isinstance(extras, dict)
    base = base_distributions()
    return {
        group: {distribution_name(requirement) for requirement in requirements} - base
        for group, requirements in extras.items()
        if group != "all"
    }
