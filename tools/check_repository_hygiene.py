"""Enforce the tracked-file repository-hygiene contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tarfile
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.release_policy import personal_paths

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "release" / "repository-hygiene-policy.json"
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".cff",
    ".cmd",
    ".css",
    ".html",
    ".ini",
    ".ipynb",
    ".js",
    ".json",
    ".md",
    ".py",
    ".pyi",
    ".ps1",
    ".rst",
    ".sh",
    ".tex",
    ".toml",
    ".ts",
    ".txt",
    ".yaml",
    ".yml",
}
EXCEPTION_FIELDS = {"path", "category", "owner", "reason", "size", "review_condition"}


@dataclass(frozen=True, slots=True)
class TrackedFile:
    path: str
    size: int
    identity: str


@dataclass(frozen=True, slots=True)
class Violation:
    code: str
    path: str
    message: str


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("schema_version") != 1:
        raise ValueError(f"Unsupported repository-hygiene policy: {path}")
    return policy


def load_exceptions(root: Path, policy: dict[str, Any]) -> list[dict[str, Any]]:
    path = root / str(policy["exception_manifest"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported repository-hygiene exception manifest: {path}")
    exceptions = list(payload.get("exceptions", []))
    for item in exceptions:
        missing = EXCEPTION_FIELDS - set(item)
        if missing:
            raise ValueError(f"Exception {item.get('path', '<unknown>')} lacks fields: {sorted(missing)}")
        exception_path = str(item["path"])
        if any(character in exception_path for character in "*?["):
            raise ValueError(f"Exception paths must be exact, not wildcards: {exception_path}")
    return exceptions


def git_tracked_files(root: Path) -> list[TrackedFile]:
    raw = _git(root, "ls-files", "--stage")
    pattern = re.compile(r"^\d+\s+(\w+)\s+0\t(.+)$")
    indexed: list[tuple[str, str]] = []
    for line in raw.splitlines():
        match = pattern.match(line)
        if not match:
            raise RuntimeError(f"Unable to parse tracked file metadata: {line!r}")
        indexed.append((match.group(1), match.group(2)))
    identities = sorted({identity for identity, _ in indexed})
    completed = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch-check=%(objectname) %(objectsize)"],
        input="\n".join(identities) + "\n",
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    sizes = {line.split()[0]: int(line.split()[1]) for line in completed.stdout.splitlines()}
    return [TrackedFile(path=path, size=sizes[identity], identity=identity) for identity, path in indexed]


def filesystem_tracked_files(root: Path, paths: list[str]) -> list[TrackedFile]:
    records = []
    for raw_path in paths:
        path = raw_path.replace("\\", "/")
        data = (root / Path(path)).read_bytes()
        records.append(TrackedFile(path=path, size=len(data), identity=hashlib.sha256(data).hexdigest()))
    return records


def _glob_regex(pattern: str) -> re.Pattern[str]:
    # Path globs: "*" and "?" stay within one directory, "**/" spans zero or more, and a trailing "**" anything below.
    parts = []
    index = 0
    while index < len(pattern):
        if pattern.startswith("**/", index):
            parts.append("(?:[^/]+/)*")
            index += 3
        elif pattern.startswith("**", index):
            parts.append(".*")
            index += 2
        elif pattern[index] == "*":
            parts.append("[^/]*")
            index += 1
        elif pattern[index] == "?":
            parts.append("[^/]")
            index += 1
        else:
            parts.append(re.escape(pattern[index]))
            index += 1
    return re.compile("".join(parts))


def _matches_any(path: str, patterns: list[str]) -> bool:
    return any(_glob_regex(pattern).fullmatch(path) for pattern in patterns)


def _text(path: Path, *, max_bytes: int = 2_000_000) -> str | None:
    if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file() or path.stat().st_size > max_bytes:
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _archive_duplicated(root: Path, path: str, tracked: set[str]) -> tuple[bool, str]:
    if not path.lower().endswith(".zip"):
        return False, ""
    candidate = PurePosixPath(path).with_suffix("")
    prefix = candidate.as_posix() + "/"
    expanded = {item[len(prefix) :]: item for item in tracked if item.startswith(prefix)}
    if not expanded:
        return False, ""
    with zipfile.ZipFile(root / Path(path)) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        names = [PurePosixPath(info.filename.replace("\\", "/")).as_posix().lstrip("./") for info in infos]
        wrappers = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
        strip_wrapper = len(wrappers) == 1 and next(iter(wrappers)) == candidate.name
        for info, name in zip(infos, names, strict=True):
            if strip_wrapper and "/" in name:
                name = name.split("/", 1)[1]
            expanded_path = expanded.get(name)
            if expanded_path is None:
                continue
            if hashlib.sha256(archive.read(info)).digest() == hashlib.sha256((root / Path(expanded_path)).read_bytes()).digest():
                return True, expanded_path
    return False, ""


def _notebook_violations(root: Path, record: TrackedFile, notebook_policy: dict[str, Any]) -> list[Violation]:
    if not record.path.endswith(".ipynb"):
        return []
    try:
        payload = json.loads((root / Path(record.path)).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [Violation("notebook_invalid", record.path, f"Notebook is not valid JSON: {exc}")]
    violations: list[Violation] = []
    cells = list(payload.get("cells", []))
    output_count = sum(len(cell.get("outputs", [])) for cell in cells)
    execution_counts = sum(cell.get("execution_count") is not None for cell in cells)
    if not notebook_policy["allow_stored_outputs"] and output_count:
        violations.append(Violation("notebook_output", record.path, f"Notebook stores {output_count} output object(s)."))
    if not notebook_policy["allow_execution_counts"] and execution_counts:
        violations.append(Violation("notebook_execution_count", record.path, f"Notebook stores {execution_counts} execution count(s)."))
    kernelspec = payload.get("metadata", {}).get("kernelspec", {})
    expected_name = notebook_policy["allowed_kernel_name"]
    expected_display = notebook_policy["allowed_display_name"]
    if kernelspec.get("name") != expected_name or kernelspec.get("display_name") != expected_display:
        violations.append(
            Violation(
                "notebook_kernelspec",
                record.path,
                f"Expected kernelspec name={expected_name!r}, display_name={expected_display!r}; got {kernelspec!r}.",
            )
        )
    return violations


def _personal_path_violation(path: str, payload: bytes) -> Violation | None:
    # The release checker's detector, on the bytes of every tracked file: no account name is exempt.
    found = personal_paths(payload)
    if not found:
        return None
    return Violation("personal_absolute_path", path, f"Absolute home-directory path {found[0]!r}; the release scan rejects it.")


def _output_declaration_violation(path: str, content: str) -> Violation | None:
    if path == "tools/check_repository_hygiene.py" or not path.startswith(("examples/", "tools/", "experiments/")):
        return None
    root_filename_default = re.search(
        r"default\s*=\s*[\"'](?!artifacts/|results/|build/|dist/)[^\"'/\\]+[.](?:csv|json|md|png|tex|pdf|zip)[\"']",
        content,
    )
    forbidden_directory_default = re.search(r"default\s*=\s*[\"'](?:reports|submission)/", content)
    cwd_open = re.search(r"(?:open|Path)\s*\(\s*[\"'](?:FUN[.]csv|X[.]csv|zdt1_table[.]tex)[\"']", content)
    if root_filename_default or forbidden_directory_default or cwd_open:
        return Violation("unsafe_output_default", path, "Producer declares a repository-root or forbidden tracked output default.")
    return None


def collect_violations(
    root: Path = ROOT,
    *,
    tracked_paths: list[str] | None = None,
    policy: dict[str, Any] | None = None,
    exceptions: list[dict[str, Any]] | None = None,
) -> list[Violation]:
    root = root.resolve()
    policy = load_policy(root / "release" / "repository-hygiene-policy.json") if policy is None else policy
    exceptions = load_exceptions(root, policy) if exceptions is None else exceptions
    records = git_tracked_files(root) if tracked_paths is None else filesystem_tracked_files(root, tracked_paths)
    record_by_path = {record.path: record for record in records}
    tracked = set(record_by_path)
    exception_by_path = {str(item["path"]): item for item in exceptions}
    violations: list[Violation] = []

    for item in exceptions:
        path = str(item["path"])
        record = record_by_path.get(path)
        if record is None:
            violations.append(Violation("stale_exception", path, "Exception path is not tracked."))
        elif int(item["size"]) != record.size:
            violations.append(
                Violation("exception_size_drift", path, f"Manifest size {item['size']} does not match tracked size {record.size}.")
            )

    root_files = dict(policy["root_files"])
    root_binary_extensions = set(policy["root_extension_rules"]["never_root_output_extensions"])
    top_level_owners = dict(policy["top_level_owners"])
    forbidden_roots = set(policy["forbidden_root_names"])
    forbidden_top_levels = set(policy["forbidden_top_level_directories"])
    temp_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in policy["temporary_name_patterns"]]
    audit_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in policy["audit_name_patterns"]]

    for record in records:
        path = record.path
        pure = PurePosixPath(path)
        disk_path = root / Path(path)
        if "/" not in path:
            if path in forbidden_roots:
                violations.append(Violation("forbidden_root_name", path, "Known generated/scratch root output is forbidden."))
            if pure.suffix.lower() in root_binary_extensions:
                violations.append(
                    Violation("root_output_file", path, "Binary or generated-output extension is forbidden at repository root.")
                )
            if path not in root_files:
                violations.append(Violation("unreviewed_root_file", path, "Root file is absent from the reviewed responsibility policy."))
            if record.size == 0:
                violations.append(Violation("empty_root_file", path, "Unexplained empty root file is forbidden."))
        else:
            top_level = path.split("/", 1)[0]
            if top_level in forbidden_top_levels:
                violations.append(Violation("forbidden_top_level", path, f"Tracked content under {top_level}/ is forbidden."))
            elif top_level not in top_level_owners:
                violations.append(Violation("unowned_top_level", path, f"Top-level directory {top_level!r} has no policy owner."))
        if any(pattern.search(path) for pattern in temp_patterns):
            violations.append(Violation("temporary_name", path, "Temporary, backup or scratch filename pattern is tracked."))
        is_audit_tool = path.startswith("tools/") and pure.suffix in {".py", ".ps1", ".sh"}
        if not is_audit_tool and any(pattern.search(path) for pattern in audit_patterns):
            violations.append(Violation("audit_evidence", path, "Raw audit/Goal handoff evidence is forbidden in the product tree."))
        if _matches_any(path, list(policy["generated_forbidden_globs"])):
            violations.append(
                Violation("generated_output", path, "Generated output is tracked in a forbidden publication/experiment location.")
            )
        if _matches_any(path, list(policy["compiled_publication_globs"])):
            violations.append(Violation("compiled_publication", path, "Compiled publication/submission output is tracked."))
        if record.size > int(policy["large_file_bytes"]):
            category = str(exception_by_path.get(path, {}).get("category", ""))
            if not category.startswith("large_"):
                violations.append(Violation("unapproved_large_file", path, f"Tracked size {record.size} exceeds policy threshold."))
        violations.extend(_notebook_violations(root, record, dict(policy["notebook"])))
        if disk_path.is_file():
            personal = _personal_path_violation(path, disk_path.read_bytes())
            if personal is not None:
                violations.append(personal)
        content = _text(disk_path)
        if content is not None:
            unsafe_output = _output_declaration_violation(path, content)
            if unsafe_output is not None:
                violations.append(unsafe_output)
        if record.size == 0 and (
            any(pattern.search(path) for pattern in temp_patterns)
            or pure.suffix.lower() in {".csv", ".json", ".log", ".png", ".sqlite", ".sqlite3", ".zip"}
        ):
            violations.append(Violation("empty_suspicious_file", path, "Empty output-like file is suspicious."))

    duplicates: dict[str, list[TrackedFile]] = defaultdict(list)
    for record in records:
        if record.size >= int(policy["duplicate_content_bytes"]):
            duplicates[record.identity].append(record)
    for group in duplicates.values():
        if len(group) < 2:
            continue
        paths = sorted(record.path for record in group)
        categories = {str(exception_by_path.get(path, {}).get("category", "")) for path in paths}
        if categories == {"semantic_duplicate_package_resource"}:
            continue
        message = f"Exact duplicate tracked content above threshold: {', '.join(paths)}"
        for path in paths:
            violations.append(Violation("exact_duplicate", path, message))

    for path in sorted(tracked):
        duplicated, counterpart = _archive_duplicated(root, path, tracked)
        if duplicated:
            violations.append(
                Violation("archive_extracted_duplicate", path, f"Archive contains an exact tracked expanded member: {counterpart}")
            )
    return sorted(set(violations), key=lambda item: (item.code, item.path, item.message))


def distribution_violations(archives: list[Path], policy: dict[str, Any] | None = None) -> list[Violation]:
    policy = load_policy() if policy is None else policy
    patterns = [str(pattern) for pattern in policy["package_forbidden_patterns"]]
    violations: list[Violation] = []
    for archive_path in archives:
        name = archive_path.name
        if name.endswith(".whl") or name.endswith(".zip"):
            with zipfile.ZipFile(archive_path) as archive:
                entries = [item for item in archive.namelist() if not item.endswith("/")]
        elif name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(archive_path, "r:gz") as archive:
                entries = [item.name for item in archive.getmembers() if item.isfile()]
        else:
            violations.append(Violation("unsupported_distribution", name, "Unsupported distribution archive type."))
            continue
        for entry in entries:
            normalized = "/" + entry.replace("\\", "/").lstrip("/")
            if any(pattern in normalized for pattern in patterns):
                violations.append(Violation("forbidden_distribution_content", name, f"Forbidden archive entry: {entry}"))
    return sorted(violations, key=lambda item: (item.path, item.message))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--artifacts", type=Path, help="Optional directory containing wheel/sdist archives to inspect.")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.root.resolve()
    policy_path = args.policy.resolve() if args.policy else root / "release" / "repository-hygiene-policy.json"
    policy = load_policy(policy_path)
    violations = collect_violations(root, policy=policy)
    if args.artifacts:
        archives = sorted(path for path in args.artifacts.iterdir() if path.is_file())
        violations.extend(distribution_violations(archives, policy))
    payload = {
        "document_type": "vamos.repository-hygiene-check",
        "schema_version": 1,
        "tracked_files": len(git_tracked_files(root)),
        "status": "failed" if violations else "passed",
        "violations": [asdict(item) for item in violations],
    }
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    elif violations:
        print(f"Repository hygiene failed with {len(violations)} violation(s):")
        for item in violations:
            print(f"- [{item.code}] {item.path}: {item.message}")
    else:
        print(f"Repository hygiene passed for {payload['tracked_files']} tracked files.")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
