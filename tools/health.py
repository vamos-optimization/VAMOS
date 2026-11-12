from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str], label: str) -> int:
    print(f"\n==> {label}")
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"FAILED: {label} (exit {result.returncode})")
        return result.returncode
    print(f"OK: {label}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run VAMOS health checks (fast-fail suite).")
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Run all checks even if one fails.",
    )
    args = parser.parse_args()

    python = sys.executable
    commands: list[tuple[str, list[str]]] = [
        # CI and the release checker run this exact no-argument command.
        ("Repository hygiene", [python, "tools/check_repository_hygiene.py"]),
        ("Canonical repository identity", [python, "tools/check_repository_identity.py"]),
        ("Repository identity tests", [python, "-m", "pytest", "-q", "tests/test_check_repository_identity.py"]),
        ("Repository hygiene checker tests", [python, "-m", "pytest", "-q", "tests/test_check_repository_hygiene.py"]),
        ("Unsupported pre-1.0 remnant audit", [python, "tools/check_pre_release_remnants.py"]),
        ("Unsupported pre-1.0 remnant checker tests", [python, "-m", "pytest", "-q", "tests/test_check_pre_release_remnants.py"]),
        # CI runs this checker with the same script and no arguments. See /AGENTS.md.
        ("Agent documentation", [python, "tools/check_agent_docs.py"]),
        ("Layer boundaries", [python, "-m", "pytest", "-q", "tests/architecture/test_layer_boundaries.py"]),
        ("Monolith guard", [python, "-m", "pytest", "-q", "tests/test_monolith_guard.py"]),
        ("Public API guard", [python, "-m", "pytest", "-q", "tests/test_public_api_guard.py"]),
        ("Import-time smoke", [python, "-m", "pytest", "-q", "tests/test_import_time_smoke.py"]),
        ("Import-time purity", [python, "-m", "pytest", "-q", "tests/architecture/test_no_import_time_side_effects.py"]),
        ("Public API snapshot", [python, "-m", "pytest", "-q", "tests/architecture/test_public_api_snapshot.py"]),
        ("Dependency policy", [python, "-m", "pytest", "-q", "tests/architecture/test_dependency_policy.py"]),
        ("No facade imports", [python, "-m", "pytest", "-q", "tests/architecture/test_no_facade_imports.py"]),
        ("Import cycles", [python, "-m", "pytest", "-q", "tests/architecture/test_experiment_import_cycles.py"]),
        ("Optional deps policy", [python, "-m", "pytest", "-q", "tests/test_optional_deps_policy.py"]),
        ("Logging policy", [python, "-m", "pytest", "-q", "tests/test_logging_policy.py"]),
        ("No prints in library", [python, "-m", "pytest", "-q", "tests/test_no_prints_in_library.py"]),
        ("No deprecation shims", [python, "-m", "pytest", "-q", "tests/test_no_deprecation_shims.py"]),
        ("Ruff lint gate", [python, "-m", "pytest", "-q", "tests/architecture/test_ruff_gate.py"]),
        ("Ruff format gate", [python, "-m", "pytest", "-q", "tests/architecture/test_ruff_format_gate.py"]),
        ("Strict typecheck", [python, "tools/typecheck.py", "--scope", "strict"]),
        ("Full typecheck", [python, "tools/typecheck.py", "--scope", "full"]),
        ("Build smoke", [python, "-m", "pytest", "-q", "tests/architecture/test_build_smoke.py"]),
        ("py.typed present", [python, "-m", "pytest", "-q", "tests/architecture/test_py_typed_present.py"]),
    ]

    failures = 0
    for label, cmd in commands:
        code = _run(cmd, label)
        if code != 0:
            failures += 1
            if not args.continue_on_failure:
                return code

    if failures:
        print(f"\nCompleted with {failures} failure(s).")
        return 1
    print("\nAll health checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
