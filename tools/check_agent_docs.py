"""Validate the active AI-agent instruction architecture and checked references."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.check_pre_release_remnants import guidance_remnant_tokens

ROOT_INSTRUCTION = "AGENTS.md"
COPILOT_ADAPTER = ".github/copilot-instructions.md"
CLAUDE_ADAPTER = "CLAUDE.md"

CANONICAL_GUIDES = (
    "CONTRIBUTING.md",
    "CODING_GUIDELINES.md",
    "docs/topics/extending.md",
    "docs/dev/architecture_health.md",
    "docs/dev/add_problem.md",
    "docs/dev/add_operator.md",
    "docs/dev/add_algorithm.md",
    "docs/dev/add_backend.md",
    "docs/dev/add_metric.md",
    "docs/dev/cli.md",
    "docs/dev/run_artifacts_and_replay.md",
    "docs/dev/studies.md",
    "docs/dev/testing.md",
    "docs/dev/typing.md",
    "docs/dev/run_artifact_contract.md",
    "docs/dev/run_artifact_acceptance_tests.md",
)

AGENT_ONLY_FORBIDDEN_PHRASES = (
    "reproduce --verify-only",
    "run_all",
    "front.csv",
    "variables.csv",
    "vamos.ux.analysis.loader",
    "foundation.core.runner",
)

DECLARATION_RE = re.compile(r"^\s*(path|symbol|cli|command):\s*(.+?)\s*$")
LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
SCOPE_RE = re.compile(r"Applies only to `([^`]+/\*\*)`\.")
MODULE_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
POSITIVE_DISCARDED_POLICY_RE = re.compile(
    r"\b(?:add|create|implement|preserve|provide|retain|support|write)\b.{0,80}"
    r"\b(?:legacy|migration|old[- ]format|compatibility[- ]only)\b",
    re.IGNORECASE,
)
POSITIVE_NONEXACT_REPLAY_RE = re.compile(
    r"\b(?:support|implement|provide|run|use)s?\b.{0,60}"
    r"\b(?:tolerant|best[-_ ]effort|cross[- ]backend|compatible)\b.{0,30}\b(?:execution|replay)\b",
    re.IGNORECASE,
)
NEGATIVE_POLICY_MARKERS = (
    "do not",
    "does not",
    "must not",
    "never",
    "not implement",
    "not implemented",
    "outside",
    "non-goal",
    "reject",
    "refuse",
    "without",
)


@dataclass(frozen=True)
class Declaration:
    source: Path
    line: int
    kind: str
    value: str


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _instruction_files(root: Path) -> list[Path]:
    files = [root / ROOT_INSTRUCTION]
    files.extend(path for path in root.rglob("AGENTS.md") if path != root / ROOT_INSTRUCTION)
    for relative in (COPILOT_ADAPTER, CLAUDE_ADAPTER):
        path = root / relative
        if path.exists():
            files.append(path)
    files.extend((root / ".github" / "instructions").glob("*.instructions.md"))
    files.extend((root / ".github" / "prompts").glob("*.prompt.md"))
    return sorted(set(files))


def _active_files(root: Path) -> list[Path]:
    files = set(_instruction_files(root))
    files.update(root / relative for relative in CANONICAL_GUIDES if (root / relative).exists())
    return sorted(files)


def _declarations(paths: list[Path]) -> list[Declaration]:
    declarations: list[Declaration] = []
    for path in paths:
        in_block = False
        for line_number, line in enumerate(_read(path).splitlines(), start=1):
            if line.strip() == "```agent-docs":
                in_block = True
                continue
            if in_block and line.strip() == "```":
                in_block = False
                continue
            if not in_block:
                continue
            match = DECLARATION_RE.match(line)
            if match is None:
                declarations.append(Declaration(path, line_number, "invalid", line.strip()))
            else:
                declarations.append(Declaration(path, line_number, match.group(1), match.group(2)))
        if in_block:
            declarations.append(Declaration(path, line_number, "invalid", "unclosed agent-docs block"))
    return declarations


def _check_root(root: Path, errors: list[str]) -> None:
    path = root / ROOT_INSTRUCTION
    if not path.is_file():
        errors.append("AGENTS.md: canonical root instruction is missing")
        return
    substantive = _substantive_line_count(_read(path))
    if not 100 <= substantive <= 220:
        errors.append(f"AGENTS.md: expected 100-220 substantive lines, found {substantive}")


def _substantive_line_count(text: str) -> int:
    count = 0
    in_checked_block = False
    for line in text.splitlines():
        if line.strip() == "```agent-docs":
            in_checked_block = True
            continue
        if in_checked_block and line.strip() == "```":
            in_checked_block = False
            continue
        if not in_checked_block and line.strip() and not line.strip().startswith("```"):
            count += 1
    return count


def _check_nested_scope(root: Path, errors: list[str]) -> None:
    for path in sorted(root.rglob("AGENTS.md")):
        if path == root / ROOT_INSTRUCTION:
            continue
        text = _read(path)
        rel = _relative(path, root)
        match = SCOPE_RE.search(text)
        if match is None:
            errors.append(f"{rel}: missing an exact 'Applies only to `<path>/**`.' scope declaration")
        else:
            declared = match.group(1)[:-3].rstrip("/")
            actual = path.parent.relative_to(root).as_posix()
            if declared != actual:
                errors.append(f"{rel}: declared scope {declared!r} does not match containing subtree {actual!r}")
        if "Inherits all repository-wide rules from `/AGENTS.md`" not in text:
            errors.append(f"{rel}: missing inheritance from /AGENTS.md")


def _check_adapters(root: Path, errors: list[str]) -> None:
    adapter_paths = [root / COPILOT_ADAPTER, root / CLAUDE_ADAPTER]
    adapter_paths.extend((root / ".github" / "instructions").glob("*.instructions.md"))
    adapter_paths.extend((root / ".github" / "prompts").glob("*.prompt.md"))
    for path in sorted(adapter_paths):
        if not path.exists():
            if path == root / COPILOT_ADAPTER:
                errors.append(f"{COPILOT_ADAPTER}: required adapter is missing")
            continue
        text = _read(path)
        rel = _relative(path, root)
        if "/AGENTS.md" not in text and "(AGENTS.md)" not in text:
            errors.append(f"{rel}: adapter does not reference root AGENTS.md")
    copilot = root / COPILOT_ADAPTER
    if copilot.exists() and _substantive_line_count(_read(copilot)) > 50:
        errors.append(f"{COPILOT_ADAPTER}: adapter exceeds 50 substantive lines")
    claude = root / CLAUDE_ADAPTER
    if claude.exists() and _substantive_line_count(_read(claude)) > 25:
        errors.append(f"{CLAUDE_ADAPTER}: adapter exceeds 25 substantive lines")


def _check_hidden_instructions(root: Path, errors: list[str]) -> None:
    hidden_root = root / ".agent"
    if not hidden_root.exists():
        return
    hidden = sorted(path for path in hidden_root.rglob("*") if path.is_file() and path.suffix.lower() in {".md", ".txt"})
    for path in hidden:
        errors.append(f"{_relative(path, root)}: hidden instruction source is not permitted")


def _global_rule_lines(text: str) -> set[str]:
    rules: set[str] = set()
    in_fence = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence or not line or line.startswith("#"):
            continue
        normalized = re.sub(r"^(?:[-*]|\d+[.)])\s+", "", line).casefold()
        if len(normalized) >= 32 and "inherits all repository-wide rules" not in normalized:
            rules.add(normalized)
    return rules


def _check_duplicate_global_bodies(root: Path, errors: list[str]) -> None:
    root_rules = _global_rule_lines(_read(root / ROOT_INSTRUCTION))
    for path in _instruction_files(root):
        if path == root / ROOT_INSTRUCTION or not path.exists():
            continue
        candidate = _global_rule_lines(_read(path))
        shared = root_rules & candidate
        if len(shared) >= 8 and len(shared) / max(1, len(candidate)) >= 0.4:
            errors.append(f"{_relative(path, root)}: duplicates the global instruction body")


def _check_links(root: Path, paths: list[Path], errors: list[str]) -> None:
    for path in paths:
        for line_number, line in enumerate(_read(path).splitlines(), start=1):
            for raw_target in LINK_RE.findall(line):
                target = raw_target.strip().split(maxsplit=1)[0].strip("<>")
                target = target.split("#", maxsplit=1)[0]
                if not target or target.startswith(("http://", "https://", "mailto:")):
                    continue
                resolved = (root / target.lstrip("/")) if target.startswith("/") else (path.parent / target)
                if not resolved.resolve().is_relative_to(root.resolve()):
                    errors.append(f"{_relative(path, root)}:{line_number}: link escapes repository: {raw_target}")
                elif not resolved.exists():
                    errors.append(f"{_relative(path, root)}:{line_number}: broken link: {raw_target}")


def _check_paths(root: Path, declarations: list[Declaration], errors: list[str]) -> None:
    for item in declarations:
        if item.kind != "path":
            continue
        candidate = Path(item.value)
        rel_source = _relative(item.source, root)
        if candidate.is_absolute() or ".." in candidate.parts:
            errors.append(f"{rel_source}:{item.line}: checked path must be repository-relative: {item.value}")
            continue
        resolved = (root / candidate).resolve()
        if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
            errors.append(f"{rel_source}:{item.line}: checked path does not exist: {item.value}")


def _python_env(root: Path) -> dict[str, str]:
    env = os.environ.copy()
    source = str(root / "src")
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source if not current else source + os.pathsep + current
    return env


def _check_symbols(root: Path, declarations: list[Declaration], errors: list[str]) -> None:
    checked: dict[str, Declaration] = {}
    for item in declarations:
        if item.kind != "symbol":
            continue
        rel_source = _relative(item.source, root)
        if ":" not in item.value:
            errors.append(f"{rel_source}:{item.line}: checked symbol must use module:qualname: {item.value}")
            continue
        module, qualname = item.value.split(":", maxsplit=1)
        if not MODULE_RE.fullmatch(module) or not MODULE_RE.fullmatch(qualname):
            errors.append(f"{rel_source}:{item.line}: invalid checked symbol syntax: {item.value}")
            continue
        checked.setdefault(item.value, item)
    if not checked:
        return
    script = """
import importlib
import json
import sys

for spec in json.loads(sys.argv[1]):
    module, qualname = spec.split(":", 1)
    try:
        obj = importlib.import_module(module)
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except Exception as exc:
        print(json.dumps([spec, type(exc).__name__, str(exc)]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(sorted(checked))],
        cwd=root,
        env=_python_env(root),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else f"exit {result.returncode}"
        errors.append(f"public symbol validation subprocess failed: {detail}")
        return
    for line in result.stdout.splitlines():
        symbol, error_type, message = json.loads(line)
        item = checked[symbol]
        errors.append(
            f"{_relative(item.source, root)}:{item.line}: checked public symbol is unavailable: {symbol} ({error_type}: {message})"
        )


def _check_cli(root: Path, declarations: list[Declaration], errors: list[str]) -> None:
    checked: dict[str, tuple[Declaration, list[str]]] = {}
    for item in declarations:
        if item.kind != "cli":
            continue
        rel_source = _relative(item.source, root)
        try:
            parts = shlex.split(item.value, posix=True)
        except ValueError as exc:
            errors.append(f"{rel_source}:{item.line}: invalid checked CLI syntax: {exc}")
            continue
        if len(parts) < 2 or parts[0] != "vamos" or "--help" not in parts:
            errors.append(f"{rel_source}:{item.line}: checked CLI must be a side-effect-free 'vamos ... --help' command: {item.value}")
            continue
        checked.setdefault(item.value, (item, parts))
    for value, (item, parts) in checked.items():
        rel_source = _relative(item.source, root)
        result = subprocess.run(
            [sys.executable, "-m", "vamos.experiment.cli.main", *parts[1:]],
            cwd=root,
            env=_python_env(root),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            detail_stream = result.stderr.strip() or result.stdout.strip()
            detail = detail_stream.splitlines()[-1] if detail_stream else f"exit {result.returncode}"
            errors.append(f"{rel_source}:{item.line}: checked CLI is not recognized: {value} ({detail})")


def _check_commands(root: Path, declarations: list[Declaration], errors: list[str]) -> None:
    for item in declarations:
        if item.kind != "command":
            continue
        rel_source = _relative(item.source, root)
        try:
            parts = shlex.split(item.value, posix=True)
        except ValueError as exc:
            errors.append(f"{rel_source}:{item.line}: invalid validation command syntax: {exc}")
            continue
        if not parts or parts[0] not in {"python", "pytest", "ruff", "mypy", "mkdocs", "git"}:
            errors.append(f"{rel_source}:{item.line}: unsupported validation command: {item.value}")
            continue
        if parts[0] == "python" and len(parts) > 1 and parts[1].endswith(".py"):
            _require_command_path(root, parts[1], item, errors)
        for token in parts[1:]:
            if token.startswith("-") or token in {"build", "check"} or "<" in token or ">" in token:
                continue
            if token.startswith(("tests/", "tools/", "docs/", "src/")):
                _require_command_path(root, token, item, errors)


def _require_command_path(root: Path, token: str, item: Declaration, errors: list[str]) -> None:
    clean = token.split("::", maxsplit=1)[0]
    if not (root / clean).exists():
        errors.append(f"{_relative(item.source, root)}:{item.line}: validation command path does not exist: {clean}")


def _check_vocabulary(root: Path, paths: list[Path], errors: list[str]) -> None:
    for path in paths:
        text = _read(path)
        rel = _relative(path, root)
        lower = text.casefold()
        for phrase in guidance_remnant_tokens(text):
            errors.append(f"{rel}: forbidden obsolete guidance token: {phrase}")
        for phrase in AGENT_ONLY_FORBIDDEN_PHRASES:
            if phrase.casefold() in lower:
                errors.append(f"{rel}: forbidden obsolete guidance token: {phrase}")
        for line_number, line in enumerate(text.splitlines(), start=1):
            lowered_line = line.casefold()
            is_negative = any(marker in lowered_line for marker in NEGATIVE_POLICY_MARKERS)
            if not is_negative and POSITIVE_DISCARDED_POLICY_RE.search(line):
                errors.append(f"{rel}:{line_number}: guidance reintroduces discarded pre-release behavior")
            if not is_negative and POSITIVE_NONEXACT_REPLAY_RE.search(line):
                errors.append(f"{rel}:{line_number}: guidance describes a non-exact replay mode as implemented")
        if re.search(r"\bblack\b", text, re.IGNORECASE) or re.search(r"(?:line.?length|format).{0,30}\b88\b", text, re.IGNORECASE):
            errors.append(f"{rel}: Black/88 guidance conflicts with Ruff/140")
        if "KernelBackend" in text and re.search(r"KernelBackend.{0,160}(fast_non_dominated_sort|crowding_distance)", text, re.DOTALL):
            errors.append(f"{rel}: obsolete KernelBackend method guidance")


def check_repository(root: Path) -> list[str]:
    """Return deterministic validation errors for *root*."""
    root = root.resolve()
    errors: list[str] = []
    _check_root(root, errors)
    instruction_files = _instruction_files(root)
    active_files = _active_files(root)
    missing_active = [path for path in active_files if not path.exists()]
    for path in missing_active:
        errors.append(f"{_relative(path, root)}: active agent-facing file is missing")
    existing = [path for path in active_files if path.exists()]
    declarations = _declarations(existing)
    for item in declarations:
        if item.kind == "invalid":
            errors.append(f"{_relative(item.source, root)}:{item.line}: invalid agent-docs declaration: {item.value}")
    _check_nested_scope(root, errors)
    _check_adapters(root, errors)
    _check_duplicate_global_bodies(root, errors)
    _check_hidden_instructions(root, errors)
    _check_links(root, existing, errors)
    _check_paths(root, declarations, errors)
    _check_symbols(root, declarations, errors)
    _check_cli(root, declarations, errors)
    _check_commands(root, declarations, errors)
    _check_vocabulary(root, existing, errors)
    if not instruction_files:
        errors.append("no active instruction files discovered")
    return sorted(set(errors))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    errors = check_repository(args.root)
    if errors:
        print("Agent-documentation check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    active_count = len(_active_files(args.root.resolve()))
    nested_count = len([path for path in args.root.resolve().rglob("AGENTS.md") if path != args.root.resolve() / ROOT_INSTRUCTION])
    print(f"Agent-documentation check passed: {active_count} active files, {nested_count} scoped AGENTS.md files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
