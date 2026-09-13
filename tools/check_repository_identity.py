"""Enforce canonical repository metadata, classified references, and publishing safety."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_REPOSITORY = "vamos-optimization/VAMOS"
CANONICAL_URL = f"https://github.com/{CANONICAL_REPOSITORY}"
DOCUMENTATION_URL = "https://vamos-optimization.org/"
CURRENT_DOCS_URL = DOCUMENTATION_URL
LEGACY_WEBSITE_URL = f"{DOCUMENTATION_URL}website/"
PAGES_URL = "https://vamos-optimization.github.io/VAMOS/"
PERSONAL_OWNER = "NicolasRodriguezUribe"
MIRROR_REPOSITORY = f"{PERSONAL_OWNER}/VAMOS"
MIRROR_DECLARATION = f"Personal mirror: [{MIRROR_REPOSITORY}](https://github.com/{MIRROR_REPOSITORY})."
REPOSITORY_GUARD = f"github.repository == '{CANONICAL_REPOSITORY}'"
SHELL_ASSERTION = f'test "$GITHUB_REPOSITORY" = "{CANONICAL_REPOSITORY}"'


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    classification: str


def classify_references(path: str, content: str) -> list[Finding]:
    """Default old-owner references to actionable; permit only exact governed uses."""
    findings = []
    for number, line in enumerate(content.splitlines(), 1):
        normalized = unquote(line).replace(r"\/", "/")
        if PERSONAL_OWNER.casefold() not in normalized.casefold():
            continue
        classification = "CHANGE_TO_CANONICAL"
        if path == "docs/project/repository-governance.md" and line == MIRROR_DECLARATION:
            classification = "INTENTIONAL_PERSONAL_MIRROR_REFERENCE"
        elif path == "tools/check_repository_identity.py" and line == f'PERSONAL_OWNER = "{PERSONAL_OWNER}"':
            classification = "SEMANTICALLY_UNRELATED"
        findings.append(Finding(path, number, classification))
    return findings


def scan_tracked_references(root: Path) -> list[Finding]:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        check=True,
    )
    findings = []
    for relative in completed.stdout.decode("utf-8").split("\0"):
        if relative and (root / relative).is_file():
            content = (root / relative).read_bytes().decode("utf-8", errors="replace")
            findings.extend(classify_references(relative, content))
    return findings


def load_yaml(root: Path, relative: str) -> dict:
    # BaseLoader reads MkDocs custom tags inertly and preserves GitHub's `on` key.
    return yaml.load((root / relative).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def publication_violations(workflow: dict) -> list[str]:
    jobs = workflow.get("jobs", {})
    recovery = jobs.get("recover-frozen-artifacts", {})
    violations = []
    if recovery.get("if") != REPOSITORY_GUARD:
        violations.append("Publication recovery must require the exact canonical repository guard.")
    scripts = [step["run"] for step in recovery.get("steps", []) if "run" in step]
    if not scripts or scripts[0].splitlines()[0] != SHELL_ASSERTION:
        violations.append("The first recovery shell command must assert the canonical repository.")
    if workflow.get("on", {}).get("push", {}).get("tags") != ["v1.0.0"]:
        violations.append("Publication must be triggered only by the official v1.0.0 tag.")

    def guarded(name: str, visiting: frozenset[str] = frozenset()) -> bool:
        if name == "recover-frozen-artifacts":
            return recovery.get("if") == REPOSITORY_GUARD
        if name in visiting or name not in jobs:
            return False
        job = jobs[name]
        # Custom status functions can run after a skipped dependency; reject them.
        if job.get("if") not in (None, REPOSITORY_GUARD):
            return False
        needs = job.get("needs", [])
        if isinstance(needs, str):
            needs = [needs]
        return bool(needs) and all(guarded(parent, visiting | {name}) for parent in needs)

    for name, job in jobs.items():
        if not guarded(name):
            violations.append(f"Publication job {name} is not transitively guarded.")
        if name in {"publish-testpypi", "publish-pypi"}:
            expected_environment = name.removeprefix("publish-")
            if job.get("environment") != expected_environment or job.get("permissions", {}).get("id-token") != "write":
                violations.append(f"{name} must retain its named OIDC environment.")
    for name in ("publish-testpypi", "publish-pypi", "github-release"):
        if name not in jobs:
            violations.append(f"Missing publication job: {name}.")
    return violations


def metadata_violations(root: Path) -> list[str]:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    expected_urls = {
        "Homepage": CANONICAL_URL,
        "Repository": CANONICAL_URL,
        "Issues": f"{CANONICAL_URL}/issues",
        "Documentation": DOCUMENTATION_URL,
    }
    violations = []
    for name, url in expected_urls.items():
        if project.get("urls", {}).get(name) != url:
            violations.append(f"Package {name} URL must be {url}.")
    citation = load_yaml(root, "CITATION.cff")
    if citation.get("url") != CANONICAL_URL:
        violations.append("CITATION must identify the canonical repository.")
    version_source = (root / "src/vamos/foundation/version.py").read_text(encoding="utf-8")
    if (
        project.get("name") != "vamos-optimization"
        or citation.get("version") != "1.0.0"
        or not re.search(
            r'__version__\s*=\s*[\'"]1\.0\.0[\'"]',
            version_source,
        )
    ):
        violations.append("The package and citation must retain vamos-optimization 1.0.0.")
    expected_site_urls = {
        "mkdocs.yml": CURRENT_DOCS_URL,
        "website/mkdocs.yml": LEGACY_WEBSITE_URL,
    }
    for relative, expected_site_url in expected_site_urls.items():
        config = load_yaml(root, relative)
        if config.get("repo_url") != CANONICAL_URL or config.get("repo_name") != CANONICAL_REPOSITORY:
            violations.append(f"{relative} must identify the canonical repository.")
        if config.get("site_url") != expected_site_url:
            violations.append(f"{relative} site_url must be {expected_site_url}.")
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    for label, suffix in (("Unreleased", "compare/v1.0.0...HEAD"), ("1.0.0", "releases/tag/v1.0.0")):
        if f"[{label}]: {CANONICAL_URL}/{suffix}" not in changelog:
            violations.append(f"CHANGELOG {label} link must use the canonical repository.")
    violations.extend(publication_violations(load_yaml(root, ".github/workflows/upload_pypi.yml")))
    pages = load_yaml(root, ".github/workflows/docs.yml")["jobs"]
    if pages["build"].get("if") != REPOSITORY_GUARD or pages["deploy"].get("needs") != "build":
        violations.append("Pages deployment must depend on the canonical repository guard.")
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    findings = scan_tracked_references(ROOT)
    violations = metadata_violations(ROOT)
    violations.extend(
        f"{item.path}:{item.line}: stale canonical owner" for item in findings if item.classification == "CHANGE_TO_CANONICAL"
    )
    report = {
        "repository": CANONICAL_REPOSITORY,
        "violations": violations,
        "classifications": dict(Counter(item.classification for item in findings)),
        "findings": [asdict(item) for item in findings],
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Canonical repository identity: {len(violations)} violation(s).")
        for violation in violations:
            print(violation)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
