"""Pure policy checks shared by the VAMOS release checker."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job
    import tomli as tomllib

OLD_INTERNAL_TAGS = tuple(f"v1.{minor}.0" for minor in range(6))
REQUIRED_DOCUMENTS = (
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    "docs/roadmap.md",
    "docs/project/known-limitations.md",
    "docs/project/release-notes-1.0.0.md",
)

_PERSONAL_PATH_PATTERNS = (
    re.compile(rb"[A-Za-z]:[\\/]+" + rb"Users[\\/]+[^\\/\s\x00]+", re.IGNORECASE),
    re.compile(rb"/" + rb"Users/[^/\s\x00]+"),
    re.compile(rb"/" + rb"home/[^/\s\x00]+"),
)
_CREDENTIAL_PATTERNS = {
    "private-key": re.compile(rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    "github-token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{36,}"),
    "openai-key": re.compile(rb"\bsk-(?:[A-Za-z0-9]{20,}|(?:proj|svcacct)-[A-Za-z0-9_-]{20,})\b"),
    "pypi-token": re.compile(rb"pypi-[A-Za-z0-9_-]{20,}"),
    "aws-access-key": re.compile(rb"AKIA[0-9A-Z]{16}"),
}


def git(root: Path, *arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    return completed.stdout.strip()


def repository_identity(root: Path, version: str, expected_branch: str | None, expected_commit: str | None) -> dict[str, Any]:
    status = git(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise AssertionError(f"Release worktree is not clean:\n{status}")
    head = git(root, "rev-parse", "HEAD")
    branch = os.environ.get("GITHUB_HEAD_REF") or git(root, "branch", "--show-current")
    if not branch and os.environ.get("GITHUB_ACTIONS") == "true" and os.environ.get("GITHUB_REF_TYPE") == "branch":
        branch = os.environ.get("GITHUB_REF_NAME", "")
    required_branch = expected_branch or f"release/{version}"
    if branch != required_branch:
        raise AssertionError(f"Expected branch {required_branch!r}, got {branch!r}.")
    required_commit = expected_commit or os.environ.get("VAMOS_RELEASE_COMMIT") or os.environ.get("GITHUB_SHA") or head
    resolved = git(root, "rev-parse", f"{required_commit}^{{commit}}")
    if resolved != head:
        raise AssertionError(f"Expected commit {resolved}, got {head}.")
    return {"branch": branch, "commit": head, "clean": True}


def version_evidence(root: Path, expected: str) -> dict[str, Any]:
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    runtime_text = (root / "src" / "vamos" / "foundation" / "version.py").read_text(encoding="utf-8")
    runtime_match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', runtime_text, flags=re.MULTILINE)
    if runtime_match is None:
        raise AssertionError("Runtime version declaration was not found.")
    runtime = runtime_match.group(1)
    dynamic = project.get("dynamic", [])
    source = pyproject.get("tool", {}).get("setuptools", {}).get("dynamic", {}).get("version", {}).get("attr")
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    release_notes = root / "docs" / "project" / f"release-notes-{expected}.md"
    checks = {
        "runtime": runtime == expected,
        "project_dynamic": "version" in dynamic and "version" not in project,
        "dynamic_source": source == "vamos.foundation.version.__version__",
        "citation": re.search(rf"(?m)^version:\s*['\"]?{re.escape(expected)}['\"]?\s*$", citation) is not None,
        "changelog": f"## [{expected}]" in changelog,
        "release_notes": release_notes.is_file() and expected in release_notes.read_text(encoding="utf-8"),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise AssertionError(f"Version consistency failed: {failed}")
    return {"version": expected, "runtime": runtime, "checks": checks}


def document_evidence(root: Path, version: str) -> dict[str, Any]:
    expected = list(REQUIRED_DOCUMENTS)
    expected[-1] = f"docs/project/release-notes-{version}.md"
    missing = [relative for relative in expected if not (root / relative).is_file()]
    if missing:
        raise AssertionError(f"Required release documents are missing: {missing}")
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    if "## [Unreleased]" not in changelog or f"## [{version}]" not in changelog:
        raise AssertionError("Changelog lacks the Unreleased or release-version section.")
    roadmap = (root / "docs" / "roadmap.md").read_text(encoding="utf-8").lower()
    limitations = (root / "docs" / "project" / "known-limitations.md").read_text(encoding="utf-8").lower()
    roadmap_terms = ("status", "motivation", "completion criterion", "dependency", "compatibility promise")
    limitation_terms = ("single-owner", "distributed", "exact replay", "bitwise", "trusted local", "typing")
    missing_terms = [term for term in roadmap_terms if term not in roadmap]
    missing_terms += [term for term in limitation_terms if term not in limitations]
    if missing_terms:
        raise AssertionError(f"Release documentation is missing required concepts: {missing_terms}")
    return {"documents": expected, "roadmap_terms": list(roadmap_terms), "limitation_terms": list(limitation_terms)}


def license_evidence(root: Path) -> dict[str, Any]:
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    checks = {
        "spdx": project.get("license") == "MIT",
        "license_file_declared": project.get("license-files") == ["LICENSE"],
        "license_file_present": (root / "LICENSE").is_file(),
        "beta_classifier": "Development Status :: 4 - Beta" in project.get("classifiers", []),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise AssertionError(f"License/package maturity metadata failed: {failed}")
    return checks


def tracked_files(root: Path) -> list[Path]:
    raw = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        check=True,
    ).stdout
    return [root / item.decode("utf-8", errors="surrogateescape") for item in raw.split(b"\0") if item]


def scan_files(paths: list[Path], *, root: Path) -> dict[str, Any]:
    personal: list[str] = []
    credentials: list[dict[str, str]] = []
    scanned = 0
    for path in paths:
        if not path.is_file():
            continue
        payload = path.read_bytes()
        scanned += 1
        label = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
        if any(pattern.search(payload) for pattern in _PERSONAL_PATH_PATTERNS):
            personal.append(label)
        for kind, pattern in _CREDENTIAL_PATTERNS.items():
            if pattern.search(payload):
                credentials.append({"path": label, "kind": kind})
    if personal:
        raise AssertionError(f"Personal absolute paths found in: {sorted(personal)}")
    if credentials:
        raise AssertionError(f"Obvious credential material found: {credentials}")
    return {"files_scanned": scanned, "personal_path_hits": 0, "credential_hits": 0}


def tag_evidence(root: Path, version: str, state: str) -> dict[str, Any]:
    head = git(root, "rev-parse", "HEAD")
    local = _tag_map(git(root, "show-ref", "--tags", "-d", check=False))
    remote_output = _remote_tags(root)
    remote = _tag_map(remote_output)
    official = f"v{version}"
    if state == "pre-normalization":
        missing_local = sorted(set(OLD_INTERNAL_TAGS) - set(local))
        missing_remote = sorted(set(OLD_INTERNAL_TAGS) - set(remote))
        if missing_local or missing_remote:
            raise AssertionError(f"Archived internal tag state changed; missing local={missing_local}, remote={missing_remote}.")
        if _peeled(local, official) == head or _peeled(remote, official) == head:
            raise AssertionError("The conflicting pre-public v1.0.0 tag already points at the candidate commit.")
    elif state == "pre-tag":
        remote_versions = sorted(tag for tag in remote if re.fullmatch(r"v\d+\.\d+\.\d+", tag))
        if remote_versions:
            raise AssertionError(f"Expected no public remote version tags; remote={remote_versions}.")
        if _peeled(local, official) == head:
            raise AssertionError(f"Local {official} already resolves to release commit {head}.")
    elif state == "pre-release":
        if re.fullmatch(r"\d+\.\d+\.\d+", version) is None:
            raise AssertionError("A public release requires a numeric major.minor.patch version.")
        if official in local or official in remote:
            raise AssertionError(f"Candidate tag {official} already exists locally or remotely.")
        previous = sorted(tag for tag in remote if re.fullmatch(r"v\d+\.\d+\.\d+", tag))
        if not previous:
            raise AssertionError("An update release requires an existing public release tag.")
        candidate_version = tuple(int(part) for part in version.split("."))
        for tag in previous:
            if tuple(int(part) for part in tag[1:].split(".")) >= candidate_version:
                raise AssertionError(f"Candidate {official} must be newer than published tag {tag}.")
            if f"{tag}^{{}}" not in remote or f"{tag}^{{}}" not in local:
                raise AssertionError(f"Published tag {tag} must be annotated and fetched locally.")
            if local[tag] != remote[tag] or _peeled(local, tag) != _peeled(remote, tag):
                raise AssertionError(f"Local and remote published tag {tag} differ.")
            ancestor = git(root, "merge-base", str(_peeled(remote, tag)), head)
            if ancestor != _peeled(remote, tag):
                raise AssertionError(f"Published release {tag} is not an ancestor of the candidate.")
    elif state == "normalized":
        local_versions = sorted(tag for tag in local if re.fullmatch(r"v\d+\.\d+\.\d+", tag))
        remote_versions = sorted(tag for tag in remote if re.fullmatch(r"v\d+\.\d+\.\d+", tag))
        if local_versions != [official] or remote_versions != [official]:
            raise AssertionError(f"Expected only {official}; local={local_versions}, remote={remote_versions}.")
        if _peeled(local, official) != head or _peeled(remote, official) != head:
            raise AssertionError(f"{official} does not resolve to release commit {head}.")
    elif state != "ignore":
        raise ValueError(f"Unknown tag state: {state}")
    return {
        "state": state,
        "head": head,
        "local": {tag: local[tag] for tag in sorted(local) if tag in OLD_INTERNAL_TAGS},
        "remote": {tag: remote[tag] for tag in sorted(remote) if tag in OLD_INTERNAL_TAGS},
    }


def _remote_tags(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-remote", "--tags", "origin"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Could not verify remote tags: {completed.stderr.strip()}")
    return completed.stdout


def _tag_map(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 2 or "refs/tags/" not in parts[1]:
            continue
        ref = parts[1].split("refs/tags/", 1)[1]
        values[ref] = parts[0]
    return values


def _peeled(tags: dict[str, str], tag: str) -> str | None:
    return tags.get(f"{tag}^{{}}", tags.get(tag))


__all__ = [
    "OLD_INTERNAL_TAGS",
    "document_evidence",
    "git",
    "license_evidence",
    "repository_identity",
    "scan_files",
    "tag_evidence",
    "tracked_files",
    "version_evidence",
]
