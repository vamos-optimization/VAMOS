"""The base dependencies are a closed set; everything else is an extra (docs/dev/architecture_health.md)."""

from __future__ import annotations

from tests.architecture._dependencies import base_distributions

# A new base dependency is a policy change: update this set and docs/dev/architecture_health.md together.
BASE_DEPENDENCIES = frozenset({"numpy", "scipy", "joblib", "typing-extensions"})


def test_base_dependencies_are_a_closed_set() -> None:
    declared = base_distributions()
    assert declared == BASE_DEPENDENCIES, (
        f"[project].dependencies must be exactly {sorted(BASE_DEPENDENCIES)}; "
        f"added: {sorted(declared - BASE_DEPENDENCIES)}, removed: {sorted(BASE_DEPENDENCIES - declared)}"
    )
