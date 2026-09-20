from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import release_policy  # noqa: E402
from release_artifacts import create_runtime_environment, inspect_distributions, write_release_manifests, write_runtime_lock  # noqa: E402
from release_policy import document_evidence, license_evidence, scan_files, version_evidence  # noqa: E402


def test_release_checker_has_human_and_single_json_inventory() -> None:
    completed = subprocess.run(
        [sys.executable, "tools/release_check.py", "--version", "1.0.0", "--list-checks", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    payload = json.loads(completed.stdout)
    assert completed.returncode == 0
    assert payload["document_type"] == "vamos.release-check-inventory"
    assert {
        "repository-identity",
        "stable-api-cli-schema-fixtures",
        "release-typing-policy",
        "complete-tests",
        "runtime-dependency-audit",
        "cyclonedx-sbom",
        "checksums-and-provenance",
    } <= set(payload["checks"])

    checker = (ROOT / "tools" / "release_check.py").read_text(encoding="utf-8")
    assert "os.path.abspath(args.typing_python)" in checker
    assert "Path(args.typing_python).resolve()" not in checker
    assert 'environment["PATH"] = os.pathsep.join(' in checker
    assert "str(self.runtime_python.parent)" in checker
    assert "dir=self.output.parent" in checker


def test_repository_identity_uses_branch_ref_for_detached_actions_checkout(
    monkeypatch,
    tmp_path: Path,
) -> None:
    commit = "a" * 40

    def fake_git(_root: Path, *arguments: str, check: bool = True) -> str:
        del check
        responses = {
            ("status", "--porcelain", "--untracked-files=all"): "",
            ("rev-parse", "HEAD"): commit,
            ("branch", "--show-current"): "",
            ("rev-parse", f"{commit}^{{commit}}"): commit,
        }
        return responses[arguments]

    monkeypatch.setattr(release_policy, "git", fake_git)
    monkeypatch.delenv("GITHUB_HEAD_REF", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_REF_TYPE", "branch")
    monkeypatch.setenv("GITHUB_REF_NAME", "main")

    identity = release_policy.repository_identity(tmp_path, "1.0.0", "main", commit)

    assert identity == {"branch": "main", "commit": commit, "clean": True}


def test_repository_identity_does_not_treat_tag_ref_as_branch(monkeypatch, tmp_path: Path) -> None:
    commit = "a" * 40

    def fake_git(_root: Path, *arguments: str, check: bool = True) -> str:
        del check
        responses = {
            ("status", "--porcelain", "--untracked-files=all"): "",
            ("rev-parse", "HEAD"): commit,
            ("branch", "--show-current"): "",
        }
        return responses[arguments]

    monkeypatch.setattr(release_policy, "git", fake_git)
    monkeypatch.delenv("GITHUB_HEAD_REF", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_REF_TYPE", "tag")
    monkeypatch.setenv("GITHUB_REF_NAME", "v1.0.0")

    try:
        release_policy.repository_identity(tmp_path, "1.0.0", "main", commit)
    except AssertionError as exc:
        assert "Expected branch 'main', got ''." in str(exc)
    else:
        raise AssertionError("A detached tag checkout was accepted as a branch.")


def test_pre_tag_state_requires_no_public_version_tag(monkeypatch, tmp_path: Path) -> None:
    commit = "a" * 40

    def fake_git(_root: Path, *arguments: str, check: bool = True) -> str:
        del check
        responses = {
            ("rev-parse", "HEAD"): commit,
            ("show-ref", "--tags", "-d"): "",
        }
        return responses[arguments]

    monkeypatch.setattr(release_policy, "git", fake_git)
    monkeypatch.setattr(release_policy, "_remote_tags", lambda _root: "")

    evidence = release_policy.tag_evidence(tmp_path, "1.0.0", "pre-tag")

    assert evidence == {"state": "pre-tag", "head": commit, "local": {}, "remote": {}}


def test_pre_tag_state_rejects_existing_remote_public_version_tag(monkeypatch, tmp_path: Path) -> None:
    commit = "a" * 40
    tag = "b" * 40

    def fake_git(_root: Path, *arguments: str, check: bool = True) -> str:
        del check
        responses = {
            ("rev-parse", "HEAD"): commit,
            ("show-ref", "--tags", "-d"): "",
        }
        return responses[arguments]

    monkeypatch.setattr(release_policy, "git", fake_git)
    monkeypatch.setattr(release_policy, "_remote_tags", lambda _root: f"{tag} refs/tags/v1.0.0")

    try:
        release_policy.tag_evidence(tmp_path, "1.0.0", "pre-tag")
    except AssertionError as exc:
        assert "Expected no public remote version tags" in str(exc)
    else:
        raise AssertionError("An existing remote public version tag was accepted before the release tag gate.")


def test_pre_tag_state_allows_archived_local_tag_away_from_candidate(monkeypatch, tmp_path: Path) -> None:
    commit = "a" * 40
    tag = "b" * 40
    old_commit = "c" * 40

    def fake_git(_root: Path, *arguments: str, check: bool = True) -> str:
        del check
        responses = {
            ("rev-parse", "HEAD"): commit,
            ("show-ref", "--tags", "-d"): (f"{tag} refs/tags/v1.0.0\n{old_commit} refs/tags/v1.0.0^{{}}"),
        }
        return responses[arguments]

    monkeypatch.setattr(release_policy, "git", fake_git)
    monkeypatch.setattr(release_policy, "_remote_tags", lambda _root: "")

    evidence = release_policy.tag_evidence(tmp_path, "1.0.0", "pre-tag")

    assert evidence["local"]["v1.0.0"] == tag


def test_pre_tag_state_rejects_local_tag_at_candidate(monkeypatch, tmp_path: Path) -> None:
    commit = "a" * 40
    tag = "b" * 40

    def fake_git(_root: Path, *arguments: str, check: bool = True) -> str:
        del check
        responses = {
            ("rev-parse", "HEAD"): commit,
            ("show-ref", "--tags", "-d"): f"{tag} refs/tags/v1.0.0\n{commit} refs/tags/v1.0.0^{{}}",
        }
        return responses[arguments]

    monkeypatch.setattr(release_policy, "git", fake_git)
    monkeypatch.setattr(release_policy, "_remote_tags", lambda _root: "")

    try:
        release_policy.tag_evidence(tmp_path, "1.0.0", "pre-tag")
    except AssertionError as exc:
        assert "already resolves to release commit" in str(exc)
    else:
        raise AssertionError("A local candidate tag was accepted before the release tag gate.")


def test_release_smoke_uses_only_stable_vamos_facade() -> None:
    source = (ROOT / "tools" / "release_smoke.py").read_text(encoding="utf-8")

    assert re.search(r"(?m)^\s*(?:from|import) vamos\.", source) is None
    assert "vamos.optimize(" in source
    assert "vamos.save_result(" in source
    assert "vamos.reproduce(" in source
    assert "vamos.create_study(" in source
    assert ".resume()" in source
    assert ".retry(failed_only=True)" in source
    assert "_network_denied()" in source
    assert 'removed_runtime_name = "Study" + "Runner"' in source
    assert '"vamos.run-manifest"' in source
    assert '"vamos.study-manifest"' in source
    assert "failure_exit_codes" in source
    assert 'shutil.which("vamos")' in source
    assert "Path(sys.executable).with_name" not in source


def test_release_workflows_are_parseable_pinned_and_cover_claimed_matrix() -> None:
    workflows = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    for path in workflows:
        assert yaml.safe_load(path.read_text(encoding="utf-8")) is not None
        for action in re.findall(r"(?m)^\s*uses:\s*([^\s#]+)", path.read_text(encoding="utf-8")):
            assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", action), f"Unpinned action in {path}: {action}"

    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "ubuntu-latest" in release and 'python: "3.10"' in release
    assert "windows-latest" in release and "macos-latest" in release
    assert "primary-maximum-compute" in release
    assert "python -m build --no-isolation --outdir candidate-dist" in release
    assert "path: ${{ runner.temp }}/candidate-dist" in release
    assert 'python -m venv "$RUNNER_TEMP/typing-venv"' in release
    assert '--typing-python "$RUNNER_TEMP/typing-venv/bin/python"' in release
    assert '-e ".[dev,docs,compute,analysis,examples,studio]"' in release
    assert '"tests/architecture/test_ruff_format_gate.py"' in (ROOT / "tools" / "release_check.py").read_text(encoding="utf-8")
    assert "release_smoke.py" in release
    assert "test_security_models.py" in release
    assert "vamos-${{ env.VAMOS_RELEASE_VERSION }}-frozen" in release
    assert "release/final-1.0.0" in release
    assert "--tag-state pre-tag" in release


def test_publication_uses_trusted_publishing_and_never_rebuilds() -> None:
    workflow = (ROOT / ".github" / "workflows" / "upload_pypi.yml").read_text(encoding="utf-8")

    assert "python -m build" not in workflow
    assert workflow.count("id-token: write") == 2
    assert "PYPI_API_TOKEN" not in workflow
    assert "test.pypi.org/legacy/" in workflow
    assert workflow.index("publish-testpypi:") < workflow.index("publish-pypi:")
    assert "testpypi_exists" in workflow
    assert "pypi-smoke:" in workflow
    assert "production-smoke.json" in workflow
    assert "https://pypi.org/simple/" in workflow
    assert "sha256sum --check SHA256SUMS" in workflow
    assert "gh release create" in workflow


def test_compute_extra_excludes_the_reviewed_distributed_advisory() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    constraints = (ROOT / "constraints" / "ci.txt").read_text(encoding="utf-8")
    build_requirements = (ROOT / "release" / "requirements-build.txt").read_text(encoding="utf-8")

    assert pyproject.count('"dask[distributed]>=2026.1.0"') == 2
    assert "dask==2026.8.0" in constraints
    assert "distributed==2026.8.0" in constraints
    assert 'requires = ["setuptools>=83", "wheel>=0.46.2"]' in pyproject
    assert pyproject.count('"setuptools>=83"') == 3
    assert pyproject.count('"wheel>=0.46.2"') == 3
    assert "setuptools==84.0.0" in constraints
    assert "wheel==0.46.2" in constraints
    assert "setuptools==84.0.0" in build_requirements
    assert "wheel==0.46.2" in build_requirements


def test_ci_constraints_retain_declared_python_310_support() -> None:
    constraints = (ROOT / "constraints" / "ci.txt").read_text(encoding="utf-8")

    assert 'ipython==8.39.0 ; python_version < "3.11"' in constraints
    assert 'ipython==9.0.2 ; python_version >= "3.11"' in constraints


def test_notebook_execution_dependencies_include_a_pinned_kernel() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    constraints = (ROOT / "constraints" / "ci.txt").read_text(encoding="utf-8")

    assert pyproject.count('"ipykernel>=7.3"') == 2
    assert "ipykernel==7.3.0" in constraints


def test_ci_constraints_use_reviewed_security_fixed_tooling() -> None:
    constraints = (ROOT / "constraints" / "ci.txt").read_text(encoding="utf-8")

    assert "pytest==9.1.1" in constraints
    assert "nbconvert==7.17.1" in constraints
    assert "panel==1.9.4" in constraints
    assert "bokeh==3.9.2" in constraints
    assert "pymdown-extensions==11.0.1" in constraints


def test_version_documents_and_license_policy(tmp_path: Path) -> None:
    (tmp_path / "src" / "vamos" / "foundation").mkdir(parents=True)
    (tmp_path / "docs" / "project").mkdir(parents=True)
    (tmp_path / "src" / "vamos" / "foundation" / "version.py").write_text('__version__ = "1.0.0"\n', encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "vamos-optimization"
dynamic = ["version"]
license = "MIT"
license-files = ["LICENSE"]
classifiers = ["Development Status :: 4 - Beta"]
[tool.setuptools.dynamic]
version = {attr = "vamos.foundation.version.__version__"}
""".strip(),
        encoding="utf-8",
    )
    (tmp_path / "CITATION.cff").write_text("version: 1.0.0\n", encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text("## [Unreleased]\n\n## [1.0.0]\n", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("MIT\n", encoding="utf-8")
    (tmp_path / "docs" / "roadmap.md").write_text(
        "status motivation completion criterion dependency; not a compatibility promise\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "project" / "known-limitations.md").write_text(
        "single-owner distributed exact replay bitwise trusted local typing\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "project" / "release-notes-1.0.0.md").write_text("VAMOS 1.0.0\n", encoding="utf-8")

    assert version_evidence(tmp_path, "1.0.0")["runtime"] == "1.0.0"
    assert license_evidence(tmp_path)["spdx"] is True
    assert len(document_evidence(tmp_path, "1.0.0")["documents"]) == 6


def test_path_and_secret_scan_detects_material_without_disclosing_it(tmp_path: Path) -> None:
    clean = tmp_path / "clean.txt"
    clean.write_text("ordinary release evidence with task-level-failure outcomes\n", encoding="utf-8")
    assert scan_files([clean], root=tmp_path)["credential_hits"] == 0

    personal = tmp_path / "personal.txt"
    personal.write_text("C:/" + "Users/example/private/project\n", encoding="utf-8")
    try:
        scan_files([personal], root=tmp_path)
    except AssertionError as exc:
        assert "personal.txt" in str(exc)
        assert "example" not in str(exc)
    else:
        raise AssertionError("Personal path was not detected.")

    credential = tmp_path / "credential.txt"
    credential.write_text("sk-" + "a" * 32, encoding="utf-8")
    try:
        scan_files([credential], root=tmp_path)
    except AssertionError as exc:
        assert "openai-key" in str(exc)
        assert "a" * 32 not in str(exc)
    else:
        raise AssertionError("Credential material was not detected.")


def test_distribution_inspection_checks_metadata_content_and_manifests(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "vamos_optimization-1.0.0-py3-none-any.whl"
    metadata = (
        b"Metadata-Version: 2.4\nName: vamos-optimization\nVersion: 1.0.0\n"
        b"License-Expression: MIT\nLicense-File: LICENSE\nRequires-Python: >=3.10\n\n"
    )
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("vamos/py.typed", b"")
        archive.writestr("vamos_optimization-1.0.0.dist-info/METADATA", metadata)
        archive.writestr(
            "vamos_optimization-1.0.0.dist-info/entry_points.txt", "[console_scripts]\nvamos = vamos.experiment.cli.main:main\n"
        )
        archive.writestr("vamos_optimization-1.0.0.dist-info/licenses/LICENSE", "MIT\n")
    sdist = dist / "vamos_optimization-1.0.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name, payload in {
            "vamos_optimization-1.0.0/PKG-INFO": metadata,
            "vamos_optimization-1.0.0/LICENSE": b"MIT\n",
            "vamos_optimization-1.0.0/README.md": b"# VAMOS\n",
            "vamos_optimization-1.0.0/pyproject.toml": b"[project]\n",
            "vamos_optimization-1.0.0/src/vamos/py.typed": b"",
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    evidence = inspect_distributions(dist, "1.0.0")
    sbom = tmp_path / "SBOM.cdx.json"
    sbom.write_text('{"bomFormat":"CycloneDX"}\n', encoding="utf-8")
    manifests = write_release_manifests(tmp_path, dist, sbom, "1.0.0", "a" * 40)

    assert evidence["metadata"]["license_expression"] == "MIT"
    assert (tmp_path / manifests["checksums"]).read_text(encoding="utf-8").count("  ") == 3
    assert json.loads((tmp_path / manifests["manifest"]).read_text(encoding="utf-8"))["source_commit"] == "a" * 40
    for name in ("manifest", "provenance"):
        payload = json.loads((tmp_path / manifests[name]).read_text(encoding="utf-8"))
        assert payload["repository"] == "vamos-optimization/VAMOS"
        assert payload["source_commit"] == "a" * 40


def test_clean_wheel_install_and_lock_ignore_checkout_metadata(monkeypatch, tmp_path: Path) -> None:
    shadow = tmp_path / "checkout"
    metadata_dir = shadow / "vamos_optimization-1.0.0.dist-info"
    metadata_dir.mkdir(parents=True)
    metadata = "Metadata-Version: 2.1\nName: vamos-optimization\nVersion: 1.0.0\n"
    (metadata_dir / "METADATA").write_text(metadata, encoding="utf-8")
    unrelated = shadow / "checkout_only-1.0.0.dist-info"
    unrelated.mkdir()
    (unrelated / "METADATA").write_text("Metadata-Version: 2.1\nName: checkout-only\nVersion: 1.0.0\n", encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(shadow))
    monkeypatch.setenv("PIP_NO_INDEX", "1")
    wheel = tmp_path / "vamos_optimization-1.0.0-py3-none-any.whl"
    dist_info = "vamos_optimization-1.0.0.dist-info"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("vamos/__init__.py", '__version__ = "1.0.0"\n')
        archive.writestr(f"{dist_info}/METADATA", metadata)
        archive.writestr(f"{dist_info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        archive.writestr(f"{dist_info}/RECORD", "")
    constraints = tmp_path / "constraints.txt"
    constraints.write_text("", encoding="utf-8")

    _, python = create_runtime_environment(tmp_path / "venv", wheel, constraints, extras="")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    installed = subprocess.run(
        [str(python), "-c", "import vamos; print(vamos.__file__)"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "site-packages" in installed.stdout
    assert str(shadow) not in installed.stdout
    lock = tmp_path / "runtime-lock.txt"
    write_runtime_lock(python, lock)
    assert "checkout-only" not in lock.read_text(encoding="utf-8")
