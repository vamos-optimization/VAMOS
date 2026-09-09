from __future__ import annotations

import json
from pathlib import Path

from tools import typecheck

ROOT = Path(__file__).resolve().parents[2]
FORMER_CI_PATHS = {
    "src/vamos/engine/algorithm/config",
    "src/vamos/engine/algorithm/registry.py",
    "src/vamos/engine/config/spec.py",
    "src/vamos/foundation/eval",
    "src/vamos/experiment/cli/common.py",
    "src/vamos/experiment/optimization_result",
    "src/vamos/experiment/unified.py",
}


def _diagnostic(
    *,
    path: str = "src/vamos/example.py",
    code: str = "type-arg",
    message: str = 'Missing type parameters for generic type "ndarray"',
    line: int = 10,
    column: int | None = 3,
    severity: str = "error",
) -> typecheck.Diagnostic:
    return typecheck.Diagnostic(
        path=path,
        error_code=code,
        normalized_message=message,
        line=line,
        column=column,
        severity=severity,
    )


def test_supported_toolchain_is_accepted() -> None:
    assert (
        typecheck.supported_version_errors(
            (3, 12),
            "1.15.0",
            "compiled",
            "4.16.0",
            (),
            (),
        )
        == []
    )


def test_toolchain_drift_is_rejected() -> None:
    errors = typecheck.supported_version_errors(
        (3, 11),
        "1.16.0",
        "interpreted",
        "4.15.0",
        ("types-example",),
        ("openai",),
    )

    assert len(errors) == 6


def test_parser_handles_windows_and_posix_paths_columns_notes_and_unicode() -> None:
    output = "\n".join(
        (
            r"C:\repo\src\vamos\a.py:10:4: error: Incompatible type “á”  [arg-type]",
            "/repo/src/vamos/b.py:20: note: Revealed type is builtins.int",
        )
    )

    diagnostics, unparsed = typecheck.parse_mypy_output(output, Path("C:/repo"))

    assert unparsed == []
    assert diagnostics[0].path == "src/vamos/a.py"
    assert diagnostics[0].column == 4
    assert diagnostics[0].error_code == "arg-type"
    assert diagnostics[0].normalized_message.endswith("“á”")
    assert diagnostics[1].path == "src/vamos/b.py"
    assert diagnostics[1].column is None
    assert diagnostics[1].severity == "note"


def test_fingerprint_ignores_location_but_not_semantic_identity() -> None:
    first = _diagnostic(line=10, column=3)
    moved = _diagnostic(line=999, column=1)
    changed = _diagnostic(message="A different diagnostic")

    assert first.fingerprint == moved.fingerprint
    assert first.fingerprint != changed.fingerprint


def test_repeated_diagnostics_are_a_multiset() -> None:
    diagnostic = _diagnostic()

    assert typecheck.diagnostic_counter([diagnostic, diagnostic])[diagnostic.fingerprint] == 2


def test_exact_structured_baseline_passes() -> None:
    diagnostics = [_diagnostic(), _diagnostic()]
    baseline = typecheck.build_baseline(diagnostics, "HEAD")

    comparison = typecheck.compare_ratchet(diagnostics, baseline)

    assert comparison.exact
    assert comparison.new == {}
    assert comparison.resolved == {}


def test_new_fingerprint_and_error_code_fail_ratchet() -> None:
    baseline = typecheck.build_baseline([_diagnostic()], "HEAD")
    introduced = _diagnostic(code="arg-type", message="New incompatibility")

    comparison = typecheck.compare_ratchet([_diagnostic(), introduced], baseline)

    assert not comparison.exact
    assert comparison.new == {introduced.fingerprint: 1}
    assert comparison.new_error_codes == ("arg-type",)


def test_increased_multiplicity_fails_ratchet() -> None:
    diagnostic = _diagnostic()
    baseline = typecheck.build_baseline([diagnostic], "HEAD")

    comparison = typecheck.compare_ratchet([diagnostic, diagnostic], baseline)

    assert comparison.increased == {diagnostic.fingerprint: 1}


def test_resolved_diagnostic_requires_baseline_reduction() -> None:
    diagnostic = _diagnostic()
    baseline = typecheck.build_baseline([diagnostic], "HEAD")

    comparison = typecheck.compare_ratchet([], baseline)

    assert comparison.resolved == {diagnostic.fingerprint: 1}


def test_zero_scopes_reject_diagnostics() -> None:
    assert typecheck.zero_scope_policy_errors("strict", [_diagnostic()]) == ["strict typing requires zero diagnostics."]
    assert typecheck.zero_scope_policy_errors("stable", [_diagnostic()]) == ["stable public API typing requires zero diagnostics."]
    assert typecheck.zero_scope_policy_errors("full-zero", [_diagnostic()]) == ["full-zero typing requires zero full-source diagnostics."]


def test_strict_scope_contains_the_complete_former_ci_inventory() -> None:
    assert FORMER_CI_PATHS <= set(typecheck.STRICT_PATHS)


def test_stable_scope_covers_every_supported_facade() -> None:
    assert set(typecheck.STABLE_API_PATHS) == {
        "src/vamos/__init__.py",
        "src/vamos/api.py",
        "src/vamos/algorithms.py",
        "src/vamos/problems.py",
        "src/vamos/run_artifacts.py",
        "src/vamos/study_artifacts.py",
    }


def test_mypy_command_is_nonincremental_and_uses_the_explicit_config() -> None:
    command = typecheck.build_mypy_command("strict")

    assert "--no-incremental" in command
    assert command[command.index("--config-file") + 1] == "pyproject.toml"
    assert set(typecheck.STRICT_PATHS) <= set(command)


def test_changed_production_file_with_debt_is_rejected() -> None:
    diagnostic = _diagnostic(path="src/vamos/changed.py")

    assert typecheck.touched_debt([diagnostic], {"src/vamos/changed.py"}) == {"src/vamos/changed.py": 1}


def test_configuration_policy_rejects_blanket_weakening(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[tool.mypy]
ignore_errors = true
ignore_missing_imports = true
disable_error_code = ['arg-type']
exclude = ['^src/']

[[tool.mypy.overrides]]
module = ['vamos.*']
ignore_missing_imports = true
""",
        encoding="utf-8",
    )

    errors = typecheck.suppression_policy_errors(tmp_path)

    assert "global ignore_errors=true is forbidden." in errors
    assert "global ignore_missing_imports=true is forbidden." in errors
    assert "global disabled mypy error codes are forbidden." in errors
    assert "mypy exclusions are limited to build artifacts; found ^src/." in errors
    assert any("current production modules" in error for error in errors)


def test_narrow_third_party_import_override_and_build_exclusions_are_allowed(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[tool.mypy]
ignore_missing_imports = false
exclude = ['build/', 'dist/']

[[tool.mypy.overrides]]
module = ['optional_provider', 'optional_provider.*']
ignore_missing_imports = true
""",
        encoding="utf-8",
    )

    assert typecheck.suppression_policy_errors(tmp_path) == []


def test_uncoded_ignore_in_changed_production_file_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.mypy]\nignore_missing_imports = false\n", encoding="utf-8")
    source = tmp_path / "src" / "vamos" / "changed.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = object()  # type: ignore\n", encoding="utf-8")

    errors = typecheck.suppression_policy_errors(tmp_path, {"src/vamos/changed.py"})

    assert any("uncoded type: ignore" in error for error in errors)


def test_baseline_metadata_detects_config_hash_drift() -> None:
    baseline = typecheck.build_baseline([_diagnostic()], "HEAD")
    baseline["environment"]["config_sha256"] = "0" * 64

    assert any("config_sha256" in error for error in typecheck.baseline_metadata_errors(baseline))


def test_metadata_only_pyproject_drift_preserves_typing_baseline() -> None:
    baseline = typecheck.load_baseline()

    assert typecheck._sha256(typecheck.CONFIG_PATH) != baseline["environment"]["config_sha256"]
    errors = typecheck.baseline_metadata_errors(baseline)
    assert not any("config_sha256" in error for error in errors)


def test_typing_neutral_drift_fails_closed_when_baseline_hash_is_not_anchored() -> None:
    baseline = typecheck.load_baseline()
    baseline["environment"]["config_sha256"] = "0" * 64

    assert not typecheck._typing_neutral_config_hash_drift(baseline)


def test_policy_hash_is_independent_of_checkout_line_endings(tmp_path: Path) -> None:
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    lf.write_bytes(b"first\nsecond\n")
    crlf.write_bytes(b"first\r\nsecond\r\n")

    assert typecheck._sha256(lf) == typecheck._sha256(crlf)


def test_committed_baseline_has_the_canonical_schema() -> None:
    baseline = json.loads(typecheck.BASELINE_PATH.read_text(encoding="utf-8"))

    assert baseline["schema_version"] == 1
    assert baseline["policy"] == "structured-diagnostic-ratchet"
    assert baseline["diagnostic_count"] == sum(item["multiplicity"] for item in baseline["diagnostics"])
    assert baseline["fingerprint_count"] == len(baseline["diagnostics"])


def test_health_and_ci_invoke_the_same_typecheck_commands_once() -> None:
    health = (ROOT / "tools" / "health.py").read_text(encoding="utf-8")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    strict_health = '[python, "tools/typecheck.py", "--scope", "strict"]'
    full_health = '[python, "tools/typecheck.py", "--scope", "full"]'

    assert health.count(strict_health) == 1
    assert health.count(full_health) == 1
    assert ci.count("run: python tools/typecheck.py --scope strict") == 1
    assert ci.count("run: python tools/typecheck.py --scope full") == 1
    assert "VAMOS_TYPECHECK_BASE:" in ci
    assert "fetch-depth: 0" in ci
    assert "mypy --config-file pyproject.toml" not in health
    assert "mypy --config-file pyproject.toml" not in ci


def test_release_workflows_require_canonical_release_policy() -> None:
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    publish = (ROOT / ".github" / "workflows" / "upload_pypi.yml").read_text(encoding="utf-8")
    checker = (ROOT / "tools" / "release_check.py").read_text(encoding="utf-8")

    assert release.count("python tools/release_check.py") == 1
    assert checker.count('[typing_python, "tools/typecheck.py", "--scope", "release"]') == 1
    assert publish.count('test "$(jq -r .status release-check-report.json)" = "passed"') == 1
