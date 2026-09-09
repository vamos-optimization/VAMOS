"""Run the canonical VAMOS strict, stable, full, or release typecheck."""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "pyproject.toml"
CONSTRAINTS_PATH = REPO_ROOT / "constraints" / "ci.txt"
BASELINE_PATH = REPO_ROOT / "typing" / "mypy-baseline.json"

SUPPORTED_PYTHON = (3, 12)
SUPPORTED_MYPY = "1.15.0"
SUPPORTED_TYPING_EXTENSIONS = "4.16.0"
EXPECTED_MYPY_BUILD = "compiled"
EXPECTED_STUB_PACKAGES: tuple[str, ...] = ()
ALLOWED_MYPY_EXCLUSIONS = frozenset({"build/", "dist/"})
TYPE_AFFECTING_OPTIONAL_DISTRIBUTIONS = (
    "anthropic",
    "bokeh",
    "configspace",
    "dask",
    "deap",
    "distributed",
    "google-genai",
    "hpbandster",
    "ipython",
    "ipywidgets",
    "jmetalpy",
    "matplotlib",
    "moocore",
    "networkx",
    "numba",
    "openai",
    "optuna",
    "pandas",
    "panel",
    "param",
    "platypus-opt",
    "plotly",
    "pygmo",
    "pymoo",
    "scikit-learn",
    "seaborn",
    "smac",
)

STRICT_PATHS = (
    "src/vamos/engine/algorithm/config",
    "src/vamos/engine/algorithm/registry.py",
    "src/vamos/engine/config/spec.py",
    "src/vamos/foundation/eval",
    "src/vamos/experiment/cli/common.py",
    "src/vamos/experiment/optimization_result",
    "src/vamos/experiment/unified.py",
)
FULL_PATHS = ("src/vamos",)
STABLE_API_PATHS = (
    "src/vamos/__init__.py",
    "src/vamos/api.py",
    "src/vamos/algorithms.py",
    "src/vamos/problems.py",
    "src/vamos/run_artifacts.py",
    "src/vamos/study_artifacts.py",
)
STABLE_MYPY_ARGS = (
    "--config-file",
    "pyproject.toml",
    "--no-pretty",
    "--no-color-output",
    "--show-column-numbers",
    "--no-incremental",
)

Scope = Literal["strict", "stable", "full", "release", "full-zero"]
ZeroScope = Literal["strict", "stable", "full-zero"]
DIAGNOSTIC_RE = re.compile(
    r"^(?P<path>.+?):(?P<line>\d+)(?::(?P<column>\d+))?: "
    r"(?P<severity>error|note|warning): (?P<message>.*?)(?:  \[(?P<code>[^\]]+)\])?$"
)
UNCODED_IGNORE_RE = re.compile(r"#\s*type:\s*ignore(?!\s*\[)")


@dataclass(frozen=True)
class Diagnostic:
    path: str
    error_code: str
    normalized_message: str
    line: int
    column: int | None
    severity: str

    @property
    def identity(self) -> dict[str, str]:
        return {
            "path": self.path,
            "error_code": self.error_code,
            "normalized_message": self.normalized_message,
        }

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(self.identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(raw).hexdigest().upper()


@dataclass(frozen=True)
class RatchetComparison:
    exact: bool
    new: dict[str, int]
    increased: dict[str, int]
    resolved: dict[str, int]
    new_error_codes: tuple[str, ...]


def _normalized_text_sha256(content: str) -> str:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest().upper()


def _sha256(path: Path) -> str:
    return _normalized_text_sha256(path.read_text(encoding="utf-8"))


def _distribution_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            versions[name.casefold()] = distribution.version
    return versions


def _mypy_build() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--version"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return "compiled" if "compiled: yes" in result.stdout else "interpreted"


def supported_version_errors(
    python_version: tuple[int, int],
    mypy_version: str | None,
    mypy_build: str,
    typing_extensions_version: str | None,
    stub_packages: Sequence[str],
    optional_distributions: Sequence[str],
) -> list[str]:
    errors: list[str] = []
    if python_version != SUPPORTED_PYTHON:
        errors.append(f"Python {SUPPORTED_PYTHON[0]}.{SUPPORTED_PYTHON[1]} is required; found {python_version[0]}.{python_version[1]}.")
    if mypy_version != SUPPORTED_MYPY:
        errors.append(f"mypy {SUPPORTED_MYPY} is required; found {mypy_version or 'not installed'}.")
    if mypy_build != EXPECTED_MYPY_BUILD:
        errors.append(f"a {EXPECTED_MYPY_BUILD} mypy build is required; found {mypy_build}.")
    if typing_extensions_version != SUPPORTED_TYPING_EXTENSIONS:
        errors.append(f"typing-extensions {SUPPORTED_TYPING_EXTENSIONS} is required; found {typing_extensions_version or 'not installed'}.")
    normalized_stubs = tuple(sorted(item.casefold() for item in stub_packages))
    if normalized_stubs != EXPECTED_STUB_PACKAGES:
        errors.append(f"stub package set must be empty; found {', '.join(normalized_stubs)}.")
    if optional_distributions:
        errors.append(
            "canonical typing excludes type-affecting optional provider distributions; found "
            + ", ".join(sorted(optional_distributions))
            + "."
        )
    return errors


def environment_errors() -> list[str]:
    versions = _distribution_versions()
    stubs = sorted(name for name in versions if name.startswith("types-") or "stub" in name)
    optional = sorted(name for name in TYPE_AFFECTING_OPTIONAL_DISTRIBUTIONS if name in versions)
    return supported_version_errors(
        sys.version_info[:2],
        versions.get("mypy"),
        _mypy_build(),
        versions.get("typing_extensions"),
        stubs,
        optional,
    )


def normalize_message(message: str) -> str:
    return " ".join(message.strip().split())


def normalize_path(raw_path: str, repo_root: Path = REPO_ROOT) -> str:
    normalized = raw_path.replace("\\", "/").removeprefix("./")
    root = repo_root.as_posix().rstrip("/")
    if normalized.casefold().startswith((root + "/").casefold()):
        normalized = normalized[len(root) + 1 :]
    marker = "/src/vamos/"
    marker_index = normalized.casefold().find(marker)
    if marker_index >= 0:
        normalized = normalized[marker_index + 1 :]
    return normalized


def parse_mypy_output(output: str, repo_root: Path = REPO_ROOT) -> tuple[list[Diagnostic], list[str]]:
    diagnostics: list[Diagnostic] = []
    unparsed: list[str] = []
    for line in output.splitlines():
        match = DIAGNOSTIC_RE.match(line)
        if match is None:
            if any(token in line for token in (": error:", ": note:", ": warning:")):
                unparsed.append(line)
            continue
        data = match.groupdict()
        diagnostics.append(
            Diagnostic(
                path=normalize_path(data["path"], repo_root),
                error_code=data["code"] or "<none>",
                normalized_message=normalize_message(data["message"]),
                line=int(data["line"]),
                column=int(data["column"]) if data["column"] else None,
                severity=data["severity"],
            )
        )
    return diagnostics, unparsed


def diagnostic_counter(diagnostics: Sequence[Diagnostic]) -> collections.Counter[str]:
    return collections.Counter(item.fingerprint for item in diagnostics if item.severity == "error")


def baseline_counter(baseline: dict[str, Any]) -> collections.Counter[str]:
    return collections.Counter({item["fingerprint"]: int(item["multiplicity"]) for item in baseline["diagnostics"]})


def compare_ratchet(diagnostics: Sequence[Diagnostic], baseline: dict[str, Any]) -> RatchetComparison:
    current = diagnostic_counter(diagnostics)
    expected = baseline_counter(baseline)
    new = current - expected
    resolved = expected - current
    increased = {fingerprint: count for fingerprint, count in new.items() if fingerprint in expected}
    identities = {item.fingerprint: item for item in diagnostics if item.severity == "error"}
    previous_codes = {item["error_code"] for item in baseline["diagnostics"]}
    new_error_codes = tuple(sorted({identities[key].error_code for key in new if identities[key].error_code not in previous_codes}))
    return RatchetComparison(
        exact=not new and not resolved,
        new=dict(sorted(new.items())),
        increased=dict(sorted(increased.items())),
        resolved=dict(sorted(resolved.items())),
        new_error_codes=new_error_codes,
    )


def _layer(path: str) -> str:
    prefix = "src/vamos/"
    if not path.startswith(prefix):
        return "non-production"
    remainder = path[len(prefix) :]
    return remainder.split("/", maxsplit=1)[0] if "/" in remainder else "package-root"


def build_baseline(diagnostics: Sequence[Diagnostic], generation_commit: str) -> dict[str, Any]:
    errors = [item for item in diagnostics if item.severity == "error"]
    grouped: dict[str, list[Diagnostic]] = collections.defaultdict(list)
    for diagnostic in errors:
        grouped[diagnostic.fingerprint].append(diagnostic)
    entries: list[dict[str, Any]] = []
    for fingerprint, group in sorted(grouped.items()):
        first = group[0]
        entries.append(
            {
                "fingerprint": fingerprint,
                **first.identity,
                "layer": _layer(first.path),
                "multiplicity": len(group),
                "locations": [
                    {"line": item.line, "column": item.column} for item in sorted(group, key=lambda item: (item.line, item.column or -1))
                ],
            }
        )
    versions = _distribution_versions()
    error_codes = collections.Counter(item.error_code for item in errors)
    layers = collections.Counter(_layer(item.path) for item in errors)
    return {
        "schema_version": 1,
        "policy": "structured-diagnostic-ratchet",
        "generation_commit": generation_commit,
        "generation_command": f"python tools/typecheck.py --scope full --update-baseline --generation-commit {generation_commit}",
        "environment": {
            "python": f"{SUPPORTED_PYTHON[0]}.{SUPPORTED_PYTHON[1]}",
            "mypy": SUPPORTED_MYPY,
            "mypy_build": EXPECTED_MYPY_BUILD,
            "typing_extensions": SUPPORTED_TYPING_EXTENSIONS,
            "stub_packages": list(EXPECTED_STUB_PACKAGES),
            "type_affecting_optional_distributions": [],
            "config_path": CONFIG_PATH.relative_to(REPO_ROOT).as_posix(),
            "config_sha256": _sha256(CONFIG_PATH),
            "constraints_path": CONSTRAINTS_PATH.relative_to(REPO_ROOT).as_posix(),
            "constraints_sha256": _sha256(CONSTRAINTS_PATH),
        },
        "scope": list(FULL_PATHS),
        "diagnostic_count": len(errors),
        "files_with_diagnostics": len({item.path for item in errors}),
        "fingerprint_count": len(entries),
        "error_code_summary": dict(sorted(error_codes.items())),
        "layer_summary": dict(sorted(layers.items())),
        "installed_mypy": versions.get("mypy"),
        "diagnostics": entries,
    }


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, Any]:
    data = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if data.get("schema_version") != 1 or data.get("policy") != "structured-diagnostic-ratchet":
        raise ValueError(f"Unsupported typing baseline schema in {path}.")
    return data


def _mypy_config_from_text(content: str) -> dict[str, Any] | None:
    try:
        parsed = tomllib.loads(content)
    except (tomllib.TOMLDecodeError, TypeError):
        return None
    tool = parsed.get("tool")
    if not isinstance(tool, dict):
        return None
    config = tool.get("mypy")
    return cast(dict[str, Any], config) if isinstance(config, dict) else None


def _typing_neutral_config_hash_drift(baseline: dict[str, Any]) -> bool:
    """Allow pyproject drift only when the stored config is anchored in history and mypy is unchanged."""
    environment = baseline.get("environment")
    if not isinstance(environment, dict):
        return False
    stored_hash = environment.get("config_sha256")
    if not isinstance(stored_hash, str):
        return False

    candidates: list[str] = []
    generation_commit = baseline.get("generation_commit")
    if isinstance(generation_commit, str) and generation_commit:
        candidates.append(generation_commit)
    configured_base = os.environ.get("VAMOS_TYPECHECK_BASE")
    if configured_base:
        candidates.append(configured_base)
    if subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--verify", "--quiet", "HEAD^{commit}"],
        capture_output=True,
        check=False,
    ).returncode == 0:
        history = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "log", "--format=%H", "--", "pyproject.toml"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if history.returncode == 0:
            candidates.extend(history.stdout.splitlines())

    current_mypy = _mypy_config_from_text(CONFIG_PATH.read_text(encoding="utf-8"))
    if current_mypy is None:
        return False
    seen: set[str] = set()
    for ref in candidates:
        if ref in seen:
            continue
        seen.add(ref)
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"{ref}:pyproject.toml"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0 or _normalized_text_sha256(result.stdout) != stored_hash:
            continue
        historical_mypy = _mypy_config_from_text(result.stdout)
        return historical_mypy is not None and historical_mypy == current_mypy
    return False


def baseline_metadata_errors(baseline: dict[str, Any]) -> list[str]:
    environment = baseline["environment"]
    expected = {
        "python": f"{SUPPORTED_PYTHON[0]}.{SUPPORTED_PYTHON[1]}",
        "mypy": SUPPORTED_MYPY,
        "mypy_build": EXPECTED_MYPY_BUILD,
        "typing_extensions": SUPPORTED_TYPING_EXTENSIONS,
        "stub_packages": list(EXPECTED_STUB_PACKAGES),
        "type_affecting_optional_distributions": [],
        "config_path": CONFIG_PATH.relative_to(REPO_ROOT).as_posix(),
        "config_sha256": _sha256(CONFIG_PATH),
        "constraints_path": CONSTRAINTS_PATH.relative_to(REPO_ROOT).as_posix(),
        "constraints_sha256": _sha256(CONSTRAINTS_PATH),
    }
    errors: list[str] = []
    for key, value in expected.items():
        actual = environment.get(key)
        if actual == value:
            continue
        if key == "config_sha256" and _typing_neutral_config_hash_drift(baseline):
            continue
        errors.append(f"baseline environment drift for {key}: expected {value!r}, found {actual!r}.")
    return errors


def build_mypy_command(scope: Scope) -> list[str]:
    if scope == "strict":
        paths = STRICT_PATHS
    elif scope == "stable":
        paths = STABLE_API_PATHS
    else:
        paths = FULL_PATHS
    return [sys.executable, "-m", "mypy", *STABLE_MYPY_ARGS, *paths]


def run_mypy(scope: Scope) -> tuple[list[str], int, str, list[Diagnostic], list[str]]:
    command = build_mypy_command(scope)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    environment["MYPY_FORCE_COLOR"] = "0"
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    diagnostics, unparsed = parse_mypy_output(result.stdout)
    return command, result.returncode, result.stdout, diagnostics, unparsed


def _git_output(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=check,
    )
    return result.stdout.strip()


def _valid_git_ref(ref: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def comparison_base() -> str | None:
    configured = os.environ.get("VAMOS_TYPECHECK_BASE")
    if configured:
        if not _valid_git_ref(configured):
            raise ValueError(f"VAMOS_TYPECHECK_BASE is not an available commit: {configured}")
        return configured
    return "HEAD^" if _valid_git_ref("HEAD^") else None


def changed_production_files(base: str | None = None) -> set[str]:
    changed: set[str] = set()
    resolved_base = base if base is not None else comparison_base()
    if resolved_base:
        changed.update(_git_output("diff", "--name-only", "--diff-filter=ACMR", resolved_base, "--", "src/vamos").splitlines())
    changed.update(_git_output("diff", "--name-only", "--diff-filter=ACMR", "--", "src/vamos").splitlines())
    changed.update(_git_output("ls-files", "--others", "--exclude-standard", "--", "src/vamos").splitlines())
    return {path.replace("\\", "/") for path in changed if path.endswith(".py")}


def suppression_policy_errors(root: Path = REPO_ROOT, changed_files: set[str] | None = None) -> list[str]:
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    mypy_config = config["tool"]["mypy"]
    errors: list[str] = []
    if mypy_config.get("ignore_errors") is True:
        errors.append("global ignore_errors=true is forbidden.")
    if mypy_config.get("ignore_missing_imports") is True:
        errors.append("global ignore_missing_imports=true is forbidden.")
    if mypy_config.get("disable_error_code"):
        errors.append("global disabled mypy error codes are forbidden.")
    exclusions = mypy_config.get("exclude", [])
    if isinstance(exclusions, str):
        exclusions = [exclusions]
    unexpected_exclusions = sorted(str(item) for item in exclusions if str(item) not in ALLOWED_MYPY_EXCLUSIONS)
    if unexpected_exclusions:
        errors.append("mypy exclusions are limited to build artifacts; found " + ", ".join(unexpected_exclusions) + ".")
    for override in config["tool"].get("mypy", {}).get("overrides", []):
        if override.get("ignore_errors") is True:
            errors.append(f"ignore_errors=true is forbidden for {override.get('module')!r}.")
        if override.get("disable_error_code"):
            errors.append(f"disabled error codes are forbidden for {override.get('module')!r}.")
        modules = override.get("module", [])
        if isinstance(modules, str):
            modules = [modules]
        if override.get("ignore_missing_imports") is True and any(
            module == "*" or module == "vamos" or module.startswith("vamos.") for module in modules
        ):
            errors.append(f"ignore_missing_imports=true is forbidden for current production modules {modules!r}.")
    for relative in sorted(changed_files or set()):
        path = root / relative
        if not path.is_file():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if UNCODED_IGNORE_RE.search(line):
                errors.append(f"{relative}:{line_number}: uncoded type: ignore is forbidden in a changed production file.")
    return errors


def touched_debt(diagnostics: Sequence[Diagnostic], changed_files: set[str]) -> dict[str, int]:
    counts = collections.Counter(item.path for item in diagnostics if item.severity == "error" and item.path in changed_files)
    return dict(sorted(counts.items()))


def zero_scope_policy_errors(scope: ZeroScope, diagnostics: Sequence[Diagnostic]) -> list[str]:
    count = sum(item.severity == "error" for item in diagnostics)
    if count == 0:
        return []
    if scope == "strict":
        return ["strict typing requires zero diagnostics."]
    if scope == "stable":
        return ["stable public API typing requires zero diagnostics."]
    return ["full-zero typing requires zero full-source diagnostics."]


def _health_result() -> dict[str, Any]:
    command = [sys.executable, "tools/health.py"]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    lines = result.stdout.splitlines()
    return {
        "command": command,
        "exit_code": result.returncode,
        "passed": result.returncode == 0,
        "output_tail": lines[-80:],
    }


def _identity_map(diagnostics: Sequence[Diagnostic]) -> dict[str, dict[str, str]]:
    return {item.fingerprint: item.identity for item in diagnostics if item.severity == "error"}


def _baseline_identity_map(baseline: dict[str, Any]) -> dict[str, dict[str, str]]:
    return {
        item["fingerprint"]: {
            "path": item["path"],
            "error_code": item["error_code"],
            "normalized_message": item["normalized_message"],
        }
        for item in baseline["diagnostics"]
    }


def _expanded_differences(counts: dict[str, int], identities: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    return [{"fingerprint": key, "multiplicity": count, **identities[key]} for key, count in sorted(counts.items())]


def _write_baseline(diagnostics: Sequence[Diagnostic], generation_commit: str) -> dict[str, Any]:
    baseline = build_baseline(diagnostics, generation_commit)
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return baseline


def _render_human(report: dict[str, Any]) -> None:
    print(f"Canonical VAMOS typecheck: {report['scope']}")
    print("Command:", " ".join(report["command"]))
    print(f"Diagnostics: {report['diagnostic_count']} in {report['files_with_diagnostics']} file(s)")
    if report.get("comparison"):
        comparison = report["comparison"]
        print(f"Ratchet: new={len(comparison['new'])}, increased={len(comparison['increased'])}, resolved={len(comparison['resolved'])}")
    if report.get("touched_debt"):
        print("Changed production files with debt:", ", ".join(report["touched_debt"]))
    for error in report["policy_errors"]:
        print(f"POLICY ERROR: {error}")
    print("PASS" if report["passed"] else "FAIL")


def _emit(report: dict[str, Any], output_format: str, report_path: Path | None) -> None:
    content = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(content, encoding="utf-8", newline="\n")
    if output_format == "json":
        print(content, end="")
    else:
        _render_human(report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("strict", "stable", "full", "release", "full-zero"), required=True)
    parser.add_argument("--format", choices=("human", "json"), default="human")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--generation-commit")
    parser.add_argument("--review-environment-change", action="store_true")
    args = parser.parse_args(argv)
    scope: Scope = args.scope

    if args.update_baseline and scope != "full":
        parser.error("--update-baseline is valid only with --scope full")
    if args.update_baseline and not args.generation_commit:
        parser.error("--update-baseline requires --generation-commit")
    if args.review_environment_change and not args.update_baseline:
        parser.error("--review-environment-change requires --update-baseline")

    policy_errors = environment_errors()
    try:
        changed_files = changed_production_files()
    except ValueError as exc:
        changed_files = set()
        policy_errors.append(str(exc))
    policy_errors.extend(suppression_policy_errors(changed_files=changed_files))
    subchecks: dict[str, Any] = {}
    if scope == "release":
        for child_scope in ("strict", "stable"):
            child_command, child_exit, _child_output, child_diagnostics, child_unparsed = run_mypy(cast(Scope, child_scope))
            child_errors = [item for item in child_diagnostics if item.severity == "error"]
            subchecks[child_scope] = {
                "command": child_command,
                "mypy_exit": child_exit,
                "diagnostic_count": len(child_errors),
                "files_with_diagnostics": len({item.path for item in child_errors}),
                "unparsed_diagnostic_lines": child_unparsed,
                "passed": child_exit == 0 and not child_errors and not child_unparsed,
            }
            if not subchecks[child_scope]["passed"]:
                policy_errors.append(f"{child_scope} typing must pass for the release policy.")
        command, mypy_exit, _output, diagnostics, unparsed = run_mypy("full")
    else:
        command, mypy_exit, _output, diagnostics, unparsed = run_mypy(scope)
    errors = [item for item in diagnostics if item.severity == "error"]
    if unparsed:
        policy_errors.append(f"mypy emitted {len(unparsed)} unparsed diagnostic line(s).")
    if mypy_exit not in {0, 1}:
        policy_errors.append(f"mypy invocation failed with unexpected exit {mypy_exit}.")

    comparison_data: dict[str, Any] | None = None
    debt = touched_debt(errors, changed_files)
    baseline: dict[str, Any] | None = None
    metadata_drift: list[str] = []

    if scope in {"full", "release"} and BASELINE_PATH.is_file():
        try:
            baseline = load_baseline()
            metadata_drift = baseline_metadata_errors(baseline)
            if not (args.update_baseline and args.review_environment_change):
                policy_errors.extend(metadata_drift)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            policy_errors.append(str(exc))
    elif scope in {"full", "release"} and not args.update_baseline:
        policy_errors.append(f"canonical structured baseline is missing: {BASELINE_PATH.relative_to(REPO_ROOT)}")

    if args.update_baseline:
        generation_commit = str(args.generation_commit)
        if not _valid_git_ref(generation_commit):
            policy_errors.append(f"generation commit does not exist: {generation_commit}")
        if baseline is not None:
            comparison = compare_ratchet(errors, baseline)
            if comparison.new:
                policy_errors.append("baseline update refused because new or increased diagnostics exist.")
            comparison_data = asdict(comparison)
        if debt:
            policy_errors.append("baseline update refused because a changed production file still contains typing debt.")
        if not policy_errors:
            strict_command, strict_exit, _strict_output, strict_diagnostics, strict_unparsed = run_mypy("strict")
            strict_errors = [item for item in strict_diagnostics if item.severity == "error"]
            if strict_exit != 0 or strict_errors or strict_unparsed:
                policy_errors.append(
                    f"baseline update refused because strict is not clean ({len(strict_errors)} diagnostics; command {' '.join(strict_command)})."
                )
        if not policy_errors:
            baseline = _write_baseline(errors, generation_commit)
    elif scope in {"full", "release"} and baseline is not None:
        comparison = compare_ratchet(errors, baseline)
        comparison_data = asdict(comparison)
        identities = _identity_map(errors)
        previous_identities = _baseline_identity_map(baseline)
        if comparison.new:
            policy_errors.append(
                "new or increased diagnostics violate the full-source ratchet: "
                + json.dumps(_expanded_differences(comparison.new, identities), ensure_ascii=False)
            )
        if comparison.resolved:
            policy_errors.append(
                "resolved diagnostics remain in the baseline; update it in this change: "
                + json.dumps(_expanded_differences(comparison.resolved, previous_identities), ensure_ascii=False)
            )
        if debt:
            policy_errors.append("changed production files must be free of baseline debt.")
    elif scope in {"strict", "stable", "full-zero"}:
        policy_errors.extend(zero_scope_policy_errors(cast(ZeroScope, scope), errors))

    health: dict[str, Any] | None = None
    if scope == "release" and not policy_errors:
        health = _health_result()
        if not health["passed"]:
            policy_errors.append("health must pass for the release policy.")

    zero_mypy_scope = scope in {"strict", "stable", "full-zero"}
    passed = not policy_errors and (mypy_exit == 0 if zero_mypy_scope else mypy_exit in {0, 1})
    report = {
        "schema_version": 1,
        "scope": scope,
        "policy": (
            "release-strict-stable-ratchet-health" if scope == "release" else "structured-diagnostic-ratchet" if scope == "full" else "zero"
        ),
        "command": command,
        "mypy_exit": mypy_exit,
        "diagnostic_count": len(errors),
        "files_with_diagnostics": len({item.path for item in errors}),
        "unparsed_diagnostic_lines": unparsed,
        "changed_production_files": sorted(changed_files),
        "touched_debt": debt,
        "comparison": comparison_data,
        "subchecks": subchecks,
        "health": health,
        "reviewed_environment_drift": metadata_drift if args.review_environment_change else [],
        "baseline": BASELINE_PATH.relative_to(REPO_ROOT).as_posix() if baseline is not None else None,
        "baseline_sha256": _sha256(BASELINE_PATH) if BASELINE_PATH.is_file() else None,
        "policy_errors": policy_errors,
        "passed": passed,
    }
    _emit(report, args.format, args.report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
