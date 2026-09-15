from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Classification(str, Enum):
    ACTIONABLE_PRE_RELEASE_REMNANT = "ACTIONABLE_PRE_RELEASE_REMNANT"
    CURRENT_CANONICAL_COMPATIBILITY = "CURRENT_CANONICAL_COMPATIBILITY"
    SUPPORTED_PLATFORM_COMPATIBILITY = "SUPPORTED_PLATFORM_COMPATIBILITY"
    SCIENTIFIC_FORMAT_COMPATIBILITY = "SCIENTIFIC_FORMAT_COMPATIBILITY"
    VENDORED_OR_EXTERNAL = "VENDORED_OR_EXTERNAL"
    CHECKER_NEGATIVE_FIXTURE = "CHECKER_NEGATIVE_FIXTURE"
    AUDIT_OR_HISTORICAL_EXTERNAL = "AUDIT_OR_HISTORICAL_EXTERNAL"
    SEMANTICALLY_UNRELATED = "SEMANTICALLY_UNRELATED"


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    signature: str
    classification: Classification = Classification.ACTIONABLE_PRE_RELEASE_REMNANT


_TEXT_SUFFIXES = {".cfg", ".ipynb", ".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
_SCAN_ROOTS = (".github", "docs", "examples", "experiments", "notebooks", "src", "submission", "tests", "website")
_SCAN_FILES = ("AGENTS.md", "CLAUDE.md", "CONTRIBUTING.md", "README.md")
_CHECKER_FIXTURES = {
    "tests/architecture/test_no_legacy_typing_hints.py",
    "tests/test_no_deprecation_shims.py",
    "tests/test_check_agent_docs.py",
    "tests/test_check_pre_release_remnants.py",
}
_FORBIDDEN_PATHS = (
    "src/vamos/engine/algorithm/components/variation",
    "src/vamos/engine/archive/bounded_archive.py",
    "src/vamos/experiment/cli/ablation_summary.py",
    "src/vamos/experiment/study/api.py",
    "src/vamos/experiment/study/persistence.py",
    "src/vamos/experiment/study/runner.py",
    "src/vamos/experiment/study/types.py",
    "website/_compat",
)
_DISCARDED_RUN_FILES = (
    re.compile(r"(?<![A-Za-z0-9_])FUN\.csv", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9_])X\.csv", re.IGNORECASE),
    re.compile(r"(?<![A-Za-z0-9_])G\.csv", re.IGNORECASE),
    re.compile(r"metadata\.json", re.IGNORECASE),
    re.compile(r"resolved_config\.json", re.IGNORECASE),
    re.compile(r"time\.txt", re.IGNORECASE),
)
_ACTIVE_SIGNATURES = (
    ("discarded variation import", re.compile(r"vamos\.engine\.algorithm\.components\.variation")),
    ("discarded archive builder", re.compile(r"build_bounded_archive_cfg")),
    ("discarded archive block", re.compile(r"archive\.bounded")),
)
_ARCHIVE_CONFIG_PATHS = {
    "src/vamos/engine/algorithm/config/base.py",
    "src/vamos/engine/archive/config.py",
    "src/vamos/engine/config/spec.py",
    "src/vamos/engine/hooks/config_parse.py",
}
_DISCARDED_ARCHIVE_FIELD = re.compile(r"\b(?:archive_type|size_cap|prune_policy|nondominated_only|hv_samples)\b")
_DISCARDED_ARCHIVE_KEY = re.compile(
    r"(?:['\"](?:archive_type|size_cap|prune_policy|nondominated_only|hv_samples)['\"]\s*:|"
    r"^\s*(?:archive_type|size_cap|prune_policy|nondominated_only|hv_samples)\s*:)",
)
_CLI_SIGNATURES = (
    re.compile(r"['\"]--help-commands['\"]"),
    re.compile(r"['\"]--quickstart['\"]"),
    re.compile(r"['\"]open_results['\"]"),
    re.compile(r"['\"]create_problem['\"]"),
    re.compile(r"['\"]self-check['\"]"),
    re.compile(r"['\"]self_check['\"]"),
)
_DISCARDED_STUDY_ENVELOPE = re.compile(r"vamos\.study-plan-result")
_DISCARDED_STUDY_SIGNATURES = (
    (
        "discarded study runtime symbol",
        re.compile(r"\b(?:CSVPersister|StudyPersister|StudyRunner|StudyTask|run_study)\b"),
    ),
    (
        "discarded study runtime import",
        re.compile(r"vamos\.experiment\.study\.(?:api|persistence|runner|types)\b"),
    ),
    (
        "discarded study-local result",
        re.compile(r"(?:experiment\.study\.(?:runner|types)\.StudyResult|from\s+vamos\.experiment\.study[^\n]*\bStudyResult\b)"),
    ),
    ("discarded study caller transition API", re.compile(r"\bstudy\.migration\b")),
)

# Agent-facing files reuse these semantic signatures instead of maintaining a
# second list of discarded pre-release paths, fields, artifacts, and commands.
_GUIDANCE_SIGNATURES = (
    ("vamos.engine.algorithm.components.variation", re.compile(r"vamos\.engine\.algorithm\.components\.variation")),
    ("build_bounded_archive_cfg", re.compile(r"\bbuild_bounded_archive_cfg\b")),
    ("archive.bounded", re.compile(r"\barchive\.bounded\b")),
    ("bounded_archive.py", re.compile(r"(?:^|[/\\])bounded_archive\.py\b")),
    ("website/_compat", re.compile(r"\bwebsite[/\\]_compat\b")),
    (
        "discarded archive field",
        re.compile(r"\b(?:archive_type|size_cap|prune_policy|nondominated_only|hv_samples)\b"),
    ),
    (
        "discarded CLI alias",
        re.compile(
            r"(?:--help-commands|--quickstart|\bvamos\s+(?:summary|open_results|create_problem|self-check|self_check|benchmark)\b)",
            re.IGNORECASE,
        ),
    ),
    ("FUN.csv", re.compile(r"(?<![A-Za-z0-9_])FUN\.csv", re.IGNORECASE)),
    ("X.csv", re.compile(r"(?<![A-Za-z0-9_])X\.csv", re.IGNORECASE)),
    ("G.csv", re.compile(r"(?<![A-Za-z0-9_])G\.csv", re.IGNORECASE)),
    ("metadata.json", re.compile(r"metadata\.json", re.IGNORECASE)),
    ("resolved_config.json", re.compile(r"resolved_config\.json", re.IGNORECASE)),
    ("time.txt", re.compile(r"time\.txt", re.IGNORECASE)),
    ("vamos.lock", re.compile(r"vamos\.lock", re.IGNORECASE)),
    ("vamos.study-plan-result", _DISCARDED_STUDY_ENVELOPE),
    *_DISCARDED_STUDY_SIGNATURES,
)


def guidance_remnant_tokens(text: str) -> tuple[str, ...]:
    """Return discarded pre-release tokens found in active agent guidance."""
    return tuple(sorted({label for label, pattern in _GUIDANCE_SIGNATURES if pattern.search(text)}))


def scan(root: Path) -> list[Finding]:
    root = root.resolve()
    findings: list[Finding] = []
    for relative in _FORBIDDEN_PATHS:
        if _forbidden_path_has_source(root / relative):
            findings.append(Finding(relative, 0, "discarded path exists"))

    for path in _iter_text_files(root):
        relative = path.relative_to(root).as_posix()
        if relative in _CHECKER_FIXTURES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if _DISCARDED_STUDY_ENVELOPE.search(line):
                findings.append(Finding(relative, line_number, "discarded study command envelope"))
            for name, pattern in _DISCARDED_STUDY_SIGNATURES:
                if pattern.search(line):
                    findings.append(Finding(relative, line_number, name))
            active_experiment = _is_active_experiment(relative)
            if active_experiment:
                for pattern in _DISCARDED_RUN_FILES:
                    if pattern.search(line):
                        findings.append(Finding(relative, line_number, f"discarded run file: {pattern.pattern}"))
            active_source = relative.startswith(("src/", "website/", "tests/")) or active_experiment
            if active_source:
                for name, pattern in _ACTIVE_SIGNATURES:
                    if pattern.search(line):
                        findings.append(Finding(relative, line_number, name))
            if relative in _ARCHIVE_CONFIG_PATHS and _DISCARDED_ARCHIVE_FIELD.search(line):
                findings.append(Finding(relative, line_number, "discarded archive field"))
            if active_experiment and _DISCARDED_ARCHIVE_KEY.search(line):
                findings.append(Finding(relative, line_number, "discarded archive field"))
            if relative == "src/vamos/experiment/cli/main.py":
                for pattern in _CLI_SIGNATURES:
                    if pattern.search(line):
                        findings.append(Finding(relative, line_number, "discarded CLI alias"))
            legacy_area = (
                (relative.startswith("src/") and path.suffix.lower() == ".py") or relative.startswith("website/") or active_experiment
            )
            if legacy_area and re.search(r"\blegacy\b", line, re.IGNORECASE):
                findings.append(Finding(relative, line_number, "active legacy marker"))
    return _deduplicate(findings)


def _iter_text_files(root: Path) -> list[Path]:
    files = [root / relative for relative in _SCAN_FILES if (root / relative).is_file()]
    for top in _SCAN_ROOTS:
        base = root / top
        if base.exists():
            files.extend(item for item in base.rglob("*") if item.is_file() and item.suffix.lower() in _TEXT_SUFFIXES)
    return sorted(set(files))


def _forbidden_path_has_source(path: Path) -> bool:
    if path.is_file():
        return True
    if not path.is_dir():
        return False
    return any(item.is_file() and "__pycache__" not in item.parts and item.suffix.lower() in _TEXT_SUFFIXES for item in path.rglob("*"))


def _is_active_experiment(relative: str) -> bool:
    if not relative.startswith("experiments/"):
        return False
    nested_active_roots = ("experiments/configs/", "experiments/rfc/", "experiments/scripts/")
    if relative.startswith(nested_active_roots):
        return True
    remainder = relative.removeprefix("experiments/")
    return "/" not in remainder and (remainder.endswith(".py") or remainder == "ARTIFACT_CONTRACT.md")


def _deduplicate(findings: list[Finding]) -> list[Finding]:
    unique = {(item.path, item.line, item.signature, item.classification): item for item in findings}
    return sorted(unique.values(), key=lambda item: (item.path, item.line, item.signature))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reject unsupported pre-1.0 compatibility and run-output remnants.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    findings = scan(args.root)
    if findings:
        for item in findings:
            location = f"{item.path}:{item.line}" if item.line else item.path
            print(f"{location}: {item.classification.value}: {item.signature}")
        print(f"Actionable unsupported pre-1.0 remnants: {len(findings)}")
        return 1
    print("Actionable unsupported pre-1.0 remnants: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
