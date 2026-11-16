"""The required ``build-wheel`` check passes only when every other job of its CI run succeeded."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def _build_wheel() -> dict:
    return yaml.load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)["jobs"]["build-wheel"]


def _posix_bash() -> str | None:
    """A bash that runs POSIX scripts; Windows' own ``bash.exe`` starts WSL instead."""
    bash = shutil.which("bash")
    if bash is None or "system32" in bash.lower() or "windowsapps" in bash.lower():
        return None
    return bash


def test_build_wheel_reads_every_job_of_its_run_whatever_happened_before() -> None:
    # Twice in the merge queue, a macOS leg cancelled before it got a runner left the earlier check, which read
    # needs.*.result under the default step condition, skipped: build-wheel passed and the queue merged without that leg.
    job = _build_wheel()
    guard = job["steps"][0]

    assert guard["if"].replace(" ", "") == "${{always()}}"
    assert job["permissions"] == {"contents": "read", "actions": "read"}
    assert guard["env"] == {"GH_TOKEN": "${{ github.token }}"}
    # Every attempt's jobs: "Re-run failed jobs" lists only the rerun jobs under the new attempt, and keeps the rest.
    assert "repos/$GITHUB_REPOSITORY/actions/runs/$GITHUB_RUN_ID/jobs?filter=all" in guard["run"]


@pytest.mark.skipif(_posix_bash() is None, reason="needs a POSIX bash")
@pytest.mark.parametrize(
    ("jobs", "passes"),
    # What `gh api ... --jq` prints for each job of every attempt: the attempt, the conclusion and the name.
    [
        ("1\tsuccess\ttyping\n1\tsuccess\ttests-full (ubuntu-latest, 3.12)\n1\tsuccess\ttests-full (macos-latest, 3.12)\n", True),
        ("1\tsuccess\ttyping\n1\tcancelled\ttests-full (macos-latest, 3.12)\n1\tsuccess\ttests-minimal (3.12)\n", False),
        ("1\tsuccess\ttyping\n1\tfailure\ttests-full (windows-latest, 3.12)\n", False),
        ("1\tsuccess\ttyping\n1\tskipped\ttests-minimal (3.12)\n", False),
        # Only build-wheel was rerun: its new attempt lists no other job, and attempt 1's results still count.
        ("1\tsuccess\ttyping\n1\tsuccess\ttests-full (macos-latest, 3.12)\n", True),
        # A rerun fixed a cancelled leg: its latest attempt counts.
        ("1\tsuccess\ttyping\n1\tcancelled\ttests-full (macos-latest, 3.12)\n2\tsuccess\ttests-full (macos-latest, 3.12)\n", True),
        # A rerun that failed again fails, even after an earlier success.
        ("1\tsuccess\ttyping\n1\tsuccess\ttests-full (macos-latest, 3.12)\n2\tfailure\ttests-full (macos-latest, 3.12)\n", False),
        ("", False),
        (None, False),  # the jobs API failed
    ],
)
def test_build_wheel_fails_unless_every_other_job_succeeded(tmp_path: Path, jobs: str | None, passes: bool) -> None:
    # A stand-in for gh prints what `gh api ... --jq` prints for the run's jobs, or fails like an API error.
    fake = tmp_path / "gh"
    fake.write_text('#!/bin/sh\n[ -n "$FAKE_GH_FAILS" ] && exit 1\nprintf "%s" "$FAKE_JOBS"\n', encoding="utf-8")
    fake.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "FAKE_JOBS": jobs or "",
        "FAKE_GH_FAILS": "1" if jobs is None else "",
        "GITHUB_REPOSITORY": "owner/repo",
        "GITHUB_RUN_ID": "1",
        "GITHUB_RUN_ATTEMPT": "1",
    }
    bash = _posix_bash()
    assert bash is not None

    # GitHub runs `shell: bash` steps as `bash --noprofile --norc -eo pipefail`.
    completed = subprocess.run(
        [bash, "--noprofile", "--norc", "-eo", "pipefail", "-c", _build_wheel()["steps"][0]["run"]],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert (completed.returncode == 0) is passes, completed.stdout + completed.stderr
