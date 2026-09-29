from __future__ import annotations

import copy
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from check_repository_identity import (  # noqa: E402
    CANONICAL_REPOSITORY,
    MIRROR_DECLARATION,
    MIRROR_REPOSITORY,
    PERSONAL_OWNER,
    REPOSITORY_GUARD,
    classify_references,
    load_yaml,
    metadata_violations,
    publication_violations,
    scan_tracked_references,
)


def test_current_metadata_and_tracked_references_are_canonical() -> None:
    assert metadata_violations(ROOT) == []
    findings = scan_tracked_references(ROOT)
    assert not [item for item in findings if item.classification == "CHANGE_TO_CANONICAL"]
    mirrors = [item for item in findings if item.classification == "INTENTIONAL_PERSONAL_MIRROR_REFERENCE"]
    assert len(mirrors) == 1


@pytest.mark.parametrize(
    "relative",
    [
        "pyproject.toml",
        "README.md",
        "CHANGELOG.md",
        "CITATION.cff",
        "mkdocs.yml",
        "docs/guide/installation.md",
        ".github/workflows/release.yml",
        "tools/release_artifacts.py",
        "paper/manuscript/main.tex",
        "notebooks/example.ipynb",
        "src/vamos/example.py",
        "examples/example.py",
    ],
)
@pytest.mark.parametrize(
    "template",
    [
        "https://github.com/{owner}/VAMOS",
        "https://api.github.com/repos/{owner}/VAMOS",
        "https://raw.githubusercontent.com/{owner}/VAMOS/main/README.md",
        "https://{owner}.github.io/VAMOS/",
        "repo={owner}%2FVAMOS",
        "Owner: {owner}",
    ],
)
def test_stale_owner_is_rejected_across_active_surfaces(relative: str, template: str) -> None:
    findings = classify_references(relative, template.format(owner=PERSONAL_OWNER.swapcase()))
    assert len(findings) == 1
    assert findings[0].classification == "CHANGE_TO_CANONICAL"


def test_mirror_exception_is_exact_and_cannot_allow_stale_links_elsewhere() -> None:
    path = "docs/project/repository-governance.md"
    findings = classify_references(path, MIRROR_DECLARATION)
    assert findings[0].classification == "INTENTIONAL_PERSONAL_MIRROR_REFERENCE"
    assert classify_references("README.md", MIRROR_DECLARATION)[0].classification == "CHANGE_TO_CANONICAL"
    assert classify_references(path, MIRROR_DECLARATION + " Official source.")[0].classification == "CHANGE_TO_CANONICAL"
    findings = classify_references(path, MIRROR_DECLARATION + f"\nSource: https://github.com/{MIRROR_REPOSITORY}")
    assert findings[1].classification == "CHANGE_TO_CANONICAL"


@pytest.mark.parametrize("repository,allowed", [(CANONICAL_REPOSITORY, True), (MIRROR_REPOSITORY, False), ("someone/VAMOS", False)])
def test_actual_job_and_shell_guards_accept_only_the_organization(repository: str, allowed: bool) -> None:
    workflow = load_yaml(ROOT, ".github/workflows/upload_pypi.yml")
    assert publication_violations(workflow) == []
    job = workflow["jobs"]["recover-frozen-artifacts"]
    equality = re.fullmatch(r"github\.repository == '([^']+)'", job["if"])
    assert equality is not None
    assert (repository == equality[1]) is allowed
    script = next(step["run"] for step in job["steps"] if "run" in step)
    bash = shutil.which("bash")
    if sys.platform == "win32":
        git = shutil.which("git")
        assert git is not None
        candidates = (parent / relative for parent in Path(git).resolve().parents for relative in ("bin/bash.exe", "usr/bin/bash.exe"))
        bash = next((str(candidate) for candidate in candidates if candidate.is_file()), None)
    assert bash is not None, "The release workflow shell is required for publication safety validation."
    environment = os.environ.copy()
    environment["GITHUB_REPOSITORY"] = repository
    completed = subprocess.run([bash, "-c", script.splitlines()[0]], env=environment, capture_output=True, check=False)
    assert (completed.returncode == 0) is allowed


@pytest.mark.parametrize("mutation", ["guard", "shell", "detached-publisher", "always", "new-oidc-job", "cycle", "environment"])
def test_checker_rejects_publication_guard_bypasses(mutation: str) -> None:
    workflow = copy.deepcopy(load_yaml(ROOT, ".github/workflows/upload_pypi.yml"))
    jobs = workflow["jobs"]
    if mutation == "guard":
        jobs["recover-frozen-artifacts"]["if"] = "true"
    elif mutation == "shell":
        step = next(step for step in jobs["recover-frozen-artifacts"]["steps"] if "run" in step)
        step["run"] = "\n".join(step["run"].splitlines()[1:])
    elif mutation == "detached-publisher":
        jobs["publish-pypi"].pop("needs")
    elif mutation == "always":
        jobs["publish-pypi"]["if"] = "always()"
    elif mutation == "new-oidc-job":
        jobs["unguarded"] = {"permissions": {"id-token": "write"}, "steps": []}
    elif mutation == "cycle":
        jobs["publish-testpypi"]["needs"] = "publish-pypi"
    else:
        jobs["publish-pypi"]["environment"] = "other"
    assert publication_violations(workflow)


def test_release_and_pages_workflows_require_organization_identity() -> None:
    release = load_yaml(ROOT, ".github/workflows/release.yml")
    assert release["jobs"]["build-distributions"]["if"].startswith(REPOSITORY_GUARD + " &&")
    pages = load_yaml(ROOT, ".github/workflows/docs.yml")
    assert pages["jobs"]["build"]["if"] == REPOSITORY_GUARD
    assert pages["jobs"]["deploy"]["needs"] == "build"
    assert "if" not in pages["jobs"]["deploy"]


def test_identity_gate_is_in_health_ci_and_release_validation() -> None:
    for relative in ("tools/health.py", "tools/release_check.py", ".github/workflows/ci.yml"):
        assert "tools/check_repository_identity.py" in (ROOT / relative).read_text(encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "tools/check_repository_identity.py", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["violations"] == []
