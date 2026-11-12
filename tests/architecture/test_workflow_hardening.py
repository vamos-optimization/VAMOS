"""The documentation and CI workflows keep secrets, inputs and tool versions under control (VR-150, VR-307, VR-317)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
CLOUDFLARE_WORKFLOWS = ("docs-cloudflare.yml", "docs-cloudflare-preview.yml")


def _workflow(name: str) -> dict:
    return yaml.load((WORKFLOWS / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _steps(workflow: dict) -> list[dict]:
    return [step for job in workflow.get("jobs", {}).values() for step in job.get("steps", [])]


@pytest.mark.parametrize("name", sorted(path.name for path in WORKFLOWS.glob("*.yml")))
def test_workflows_do_not_interpolate_inputs_in_run(name: str) -> None:
    # An expression inside a script is pasted into the shell before it runs; pass it through env instead.
    interpolated = [
        step.get("name", "?") for step in _steps(_workflow(name)) if re.search(r"\$\{\{\s*(inputs|github\.event)\.", step.get("run", ""))
    ]
    assert interpolated == []


@pytest.mark.parametrize("name", CLOUDFLARE_WORKFLOWS)
def test_cloudflare_token_is_scoped_to_deploy_steps(name: str) -> None:
    workflow = _workflow(name)
    assert "CLOUDFLARE_API_TOKEN" not in json.dumps(workflow.get("env", {}))
    for job in workflow["jobs"].values():
        assert "CLOUDFLARE_API_TOKEN" not in json.dumps(job.get("env", {}))
    holders = [step for step in _steps(workflow) if "secrets.CLOUDFLARE_API_TOKEN" in json.dumps(step.get("env", {}))]
    assert holders
    for step in holders:
        # Only Wrangler itself, or the check that the secret is configured, sees the token.
        assert "wrangler" in step.get("run", "") or "credentials" in step["name"].lower(), step["name"]


@pytest.mark.parametrize("name", ["ci.yml", "performance.yml"])
def test_ci_workflows_declare_read_permissions(name: str) -> None:
    assert _workflow(name).get("permissions") == {"contents": "read"}


def test_preview_publish_prints_wrangler_output_on_failure() -> None:
    scripts = [step["run"] for step in _steps(_workflow("docs-cloudflare-preview.yml")) if "wrangler" in step.get("run", "")]
    assert scripts
    for script in scripts:
        assert "/dev/null" not in script
        captures = [line.strip() for line in script.splitlines() if "=$(" in line and "wrangler" in line]
        # Under bash -e, a failing `output=$(...)` exits before anything prints it.
        assert all(line.startswith("if ! ") for line in captures), captures


def test_wrangler_comes_from_the_lockfile() -> None:
    package = json.loads((ROOT / "cloudflare" / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((ROOT / "cloudflare" / "package-lock.json").read_text(encoding="utf-8"))
    version = package["devDependencies"]["wrangler"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", version)
    assert lock["packages"]["node_modules/wrangler"]["version"] == version
    for name in CLOUDFLARE_WORKFLOWS:
        text = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert "npx --yes" not in text
        assert "WRANGLER_VERSION" not in text
        assert "npm ci" in text
