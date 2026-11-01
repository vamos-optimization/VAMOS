from __future__ import annotations

import dataclasses
import importlib
import json
import shutil
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.engine.algorithm.config import NSGAIIConfig
from vamos.experiment.artifacts.comparison import compare_array_collections, comparisons_are_exact
from vamos.experiment.artifacts.errors import (
    ArtifactMissingError,
    ArtifactResourceLimitError,
    ComponentNotReconstructableError,
    EnvironmentIncompatibilityError,
    OutputCollisionError,
    ReplayExecutionError,
    ReplayResultMismatchError,
    ReplayUnavailableError,
    ResolvedSpecMismatchError,
    UnsupportedReplayProviderError,
)
from vamos.experiment.artifacts.jsonio import canonical_json_bytes, manifest_self_hash, sha256_bytes, sha256_file, stored_json_bytes
from vamos.experiment.artifacts.reports import ReplayReport, VerificationReport


def _save_source(path: Path, *, seed: int | None = 7, config: Any | None = None) -> vamos.StoredRun:
    result = vamos.optimize(
        "zdt1",
        algorithm="nsgaii",
        algorithm_config=config,
        termination=("max_evaluations", 16) if config is not None else None,
        pop_size=None if config is not None else 8,
        max_evaluations=None if config is not None else 16,
        engine="numpy",
        seed=seed,
        n_var=6,
    )
    return vamos.save_result(result, path)


def _read_manifest(run: Path) -> dict[str, Any]:
    return json.loads((run / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(run: Path, manifest: dict[str, Any]) -> None:
    manifest["integrity"]["manifest_sha256"] = manifest_self_hash(manifest)
    (run / "manifest.json").write_bytes(stored_json_bytes(manifest))


def _rewrite_environment(run: Path, mutate: Any) -> None:
    environment_path = run / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8"))
    mutate(environment)
    environment_path.write_bytes(stored_json_bytes(environment))
    manifest = _read_manifest(run)
    descriptor = next(item for item in manifest["artifacts"] if item["role"] == "environment")
    descriptor["bytes"] = environment_path.stat().st_size
    descriptor["sha256"] = sha256_file(environment_path)
    _write_manifest(run, manifest)


def _rewrite_resolved(run: Path, mutate: Any) -> None:
    manifest = _read_manifest(run)
    mutate(manifest["resolved_spec"])
    manifest["task_id"] = "sha256:" + sha256_bytes(canonical_json_bytes(manifest["resolved_spec"]))
    _write_manifest(run, manifest)


def _snapshot(run: Path) -> dict[str, bytes]:
    return {item.relative_to(run).as_posix(): item.read_bytes() for item in sorted(run.rglob("*")) if item.is_file()}


def test_verify_run_reports_independent_exact_dimensions(tmp_path: Path) -> None:
    source = _save_source(tmp_path / "source")

    report = vamos.verify_run(source.root, require_level="exact")

    assert isinstance(report, VerificationReport)
    assert report.artifact_integrity == "valid"
    assert report.path_safety == "valid"
    assert report.numerical_bundle_safety == "valid"
    assert report.environment.level == "exact"
    assert report.component_reconstructability == "reconstructable"
    assert report.effective_replayability == "exact"
    assert report.optimization_executed is False
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.status = "changed"  # type: ignore[misc]


def test_verify_run_environment_mismatch_and_requirement(tmp_path: Path) -> None:
    source = _save_source(tmp_path / "source")
    _rewrite_environment(source.root, lambda value: value["python"].update({"version": "0.0"}))

    report = vamos.verify_run(source.root)

    assert report.environment.level == "compatible"
    assert report.effective_replayability == "compatible"
    assert any(item.field == "$.environment.python.version" and item.blocks_exact for item in report.environment.findings)
    with pytest.raises(EnvironmentIncompatibilityError) as caught:
        vamos.verify_run(source.root, require_level="exact")
    assert caught.value.optimization_executed is False


def test_verify_run_missing_material_evidence_is_unavailable(tmp_path: Path) -> None:
    source = _save_source(tmp_path / "source")
    _rewrite_environment(source.root, lambda value: value["packages"].pop("scipy"))

    report = vamos.verify_run(source.root)

    assert report.environment.level == "unavailable"
    assert report.effective_replayability == "unavailable"


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("$.environment.packages.numpy", lambda value: value["packages"].update({"numpy": "0.0"})),
        ("$.environment.packages.scipy", lambda value: value["packages"].update({"scipy": "0.0"})),
        ("$.environment.backend.name", lambda value: value["backend"].update({"name": "different"})),
        ("$.environment.backend.package", lambda value: value["backend"].update({"package": {"name": "numpy", "version": "0.0"}})),
        ("$.environment.blas", lambda value: value.update({"blas": {"vendor": "different", "integer_width": 32}})),
        ("$.environment.threads", lambda value: value.update({"threads": {"OMP_NUM_THREADS": "999"}})),
    ],
)
def test_verify_run_material_environment_mismatches(
    field: str,
    mutate: Any,
    tmp_path: Path,
) -> None:
    source = _save_source(tmp_path / "source")
    _rewrite_environment(source.root, mutate)

    report = vamos.verify_run(source.root)

    assert report.environment.level == "compatible"
    assert any(item.field == field and item.blocks_exact for item in report.environment.findings)


def test_verify_run_implementation_fingerprint_mismatch(tmp_path: Path) -> None:
    source = _save_source(tmp_path / "source")
    manifest = _read_manifest(source.root)
    manifest["provenance"]["implementation"]["distribution"]["sha256"] = "0" * 64
    _write_manifest(source.root, manifest)

    report = vamos.verify_run(source.root)

    assert report.environment.level == "compatible"
    assert any(
        item.field == "$.provenance.implementation.distribution.sha256" and item.blocks_exact for item in report.environment.findings
    )


def test_verify_run_detects_missing_artifact(tmp_path: Path) -> None:
    source = _save_source(tmp_path / "source")
    (source.root / "result.npz").unlink()

    with pytest.raises(ArtifactMissingError):
        vamos.verify_run(source.root)


def test_verify_run_enforces_resource_limits(tmp_path: Path) -> None:
    source = _save_source(tmp_path / "source")

    with pytest.raises(ArtifactResourceLimitError) as caught:
        vamos.verify_run(source.root, limits=vamos.LoadLimits(max_artifact_bytes=1))

    assert caught.value.optimization_executed is False


def test_verification_does_not_execute_or_discover_plugins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _save_source(tmp_path / "source")
    optimize_module = importlib.import_module("vamos.experiment.optimize")
    registry_module = importlib.import_module("vamos.engine.algorithm.registry")
    provenance_module = importlib.import_module("vamos.experiment.artifacts.provenance")
    monkeypatch.setattr(optimize_module, "_run_config", lambda *args, **kwargs: pytest.fail("optimization executed"))
    monkeypatch.setattr(registry_module, "_load_algorithm_plugins", lambda *args, **kwargs: pytest.fail("plugin resolution"))
    monkeypatch.setattr(provenance_module.subprocess, "run", lambda *args, **kwargs: pytest.fail("shell command"))

    assert vamos.verify_run(source.root).effective_replayability == "exact"


_PROVENANCE = importlib.import_module("vamos.experiment.artifacts.provenance")
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(root: Path, *args: str) -> str:
    identity = ("-c", "user.name=VAMOS tests", "-c", "user.email=tests@example.invalid", "-c", "commit.gpgsign=false")
    completed = subprocess.run(["git", *identity, *args], cwd=root, check=True, capture_output=True, text=True)
    return completed.stdout.strip()


def _commit_all(root: Path, message: str) -> str:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", message)
    return _git(root, "rev-parse", "HEAD")


class _InstalledVamos:
    """Installed ``vamos-optimization`` metadata of a wheel or of an editable install."""

    files = None

    def __init__(self, *, editable: bool) -> None:
        self._direct_url = json.dumps({"dir_info": {"editable": True}, "url": "file:///vamos"}) if editable else None

    def read_text(self, name: str) -> str | None:
        return self._direct_url if name == "direct_url.json" else None


class _Metadata:
    def __init__(self, installed: _InstalledVamos) -> None:
        self._installed = installed

    def distribution(self, name: str) -> Any:
        return self._installed if name == "vamos-optimization" else importlib_metadata.distribution(name)

    def __getattr__(self, name: str) -> Any:
        return getattr(importlib_metadata, name)


def _relocate_vamos(monkeypatch: pytest.MonkeyPatch, package_root: Path, *, editable: bool) -> Path:
    """Make provenance see the imported ``vamos`` package at ``package_root``."""
    module_path = package_root / "experiment" / "artifacts" / "provenance.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text("# stand-in for the imported module\n", encoding="utf-8")
    monkeypatch.setattr(_PROVENANCE, "__file__", str(module_path))
    monkeypatch.setattr(_PROVENANCE, "importlib_metadata", _Metadata(_InstalledVamos(editable=editable)))
    return package_root


def _blocking_fields(report: VerificationReport) -> list[str]:
    return [item.field for item in report.environment.findings if item.blocks_exact]


@needs_git
def test_wheel_inside_user_repository_records_wheel_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # A wheel installed in a .venv inside the user's own Git project: the project is not a VAMOS checkout.
    project = tmp_path / "project"
    _relocate_vamos(monkeypatch, project / ".venv" / "Lib" / "site-packages" / "vamos", editable=False)
    (project / ".gitignore").write_text(".venv/\n", encoding="utf-8")
    (project / "analysis.py").write_text("print('user code')\n", encoding="utf-8")
    _git(project, "init", "-q")
    _commit_all(project, "user project")
    commands: list[Any] = []
    real_run = subprocess.run

    def recording_run(*args: Any, **kwargs: Any) -> Any:
        commands.append(args[0] if args else kwargs.get("args"))
        return real_run(*args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(_PROVENANCE.subprocess, "run", recording_run)
        source = _save_source(tmp_path / "run")

    stored = _read_manifest(source.root)["provenance"]["source"]
    assert stored["kind"] == "wheel"
    assert stored["git_sha"] is None
    assert not [command for command in commands if command and command[0] == "git"]

    (project / "notes.txt").write_text("unrelated\n", encoding="utf-8")
    _commit_all(project, "unrelated user commit")
    report = vamos.verify_run(source.root)
    assert report.effective_replayability == "exact", _blocking_fields(report)


@needs_git
def test_vamos_checkout_records_checkout_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkout = tmp_path / "VAMOS"
    _relocate_vamos(monkeypatch, checkout / "src" / "vamos", editable=False)
    _git(checkout, "init", "-q")
    head = _commit_all(checkout, "checkout")

    source = _save_source(tmp_path / "run")

    stored = _read_manifest(source.root)["provenance"]["source"]
    assert stored["kind"] == "checkout"
    assert stored["git_sha"] == head
    assert stored["dirty"] is False
    report = vamos.verify_run(source.root)
    assert report.effective_replayability == "exact", _blocking_fields(report)


def test_editable_install_without_git_round_trips_exact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # For example, a "Download ZIP" source tree installed with ``pip install -e .``.
    package = tmp_path / "download" / "src" / "vamos"
    if _PROVENANCE._find_git_root(package / "__init__.py") is not None:
        pytest.skip("the temporary directory is inside a Git repository")
    _relocate_vamos(monkeypatch, package, editable=True)

    source = _save_source(tmp_path / "run")

    assert _read_manifest(source.root)["provenance"]["source"]["kind"] == "checkout"
    report = vamos.verify_run(source.root)
    assert report.effective_replayability == "exact", _blocking_fields(report)


@needs_git
def test_modified_package_data_blocks_exact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    checkout = tmp_path / "VAMOS"
    package = _relocate_vamos(monkeypatch, checkout / "src" / "vamos", editable=False)
    data = package / "resources" / "tsplib" / "demo.tsp"
    data.parent.mkdir(parents=True)
    data.write_text("1 1380 939\n", encoding="utf-8")
    _git(checkout, "init", "-q")
    _commit_all(checkout, "checkout")
    source = _save_source(tmp_path / "run")
    assert vamos.verify_run(source.root).effective_replayability == "exact"

    data.write_text("1 1381 939\n", encoding="utf-8")

    report = vamos.verify_run(source.root)
    assert report.effective_replayability == "compatible"
    assert "$.provenance.source.package_data_sha256" in _blocking_fields(report)


def test_malicious_component_string_is_inert(tmp_path: Path) -> None:
    source = _save_source(tmp_path / "source")
    marker = "manifest_owned_module_that_must_not_import"

    def mutate(resolved: dict[str, Any]) -> None:
        resolved["algorithm"]["component_id"] = f"{marker}:payload@1"
        resolved["algorithm"]["provider"] = {"type": "plugin", "distribution": marker}
        resolved["algorithm"]["config"]["module"] = marker

    _rewrite_resolved(source.root, mutate)

    summary = importlib.import_module("vamos.experiment.artifacts.inspection").inspect_run(source.root)
    report = vamos.verify_run(source.root)

    assert summary["algorithm"] == "payload"
    assert report.component_reconstructability == "unavailable"
    assert marker not in sys.modules


@pytest.mark.parametrize(
    ("expected_status", "expected_error", "mutate"),
    [
        (
            "unavailable",
            ComponentNotReconstructableError,
            lambda resolved: resolved["algorithm"].update({"component_id": "vamos.algorithm:unknown@1"}),
        ),
        (
            "unavailable",
            ComponentNotReconstructableError,
            lambda resolved: resolved["operators"]["crossover"].update({"component_id": "vamos.operator:unknown@1"}),
        ),
        (
            "unavailable",
            UnsupportedReplayProviderError,
            lambda resolved: resolved["algorithm"].update({"provider": {"type": "plugin", "distribution": "external"}}),
        ),
        (
            "manual",
            UnsupportedReplayProviderError,
            lambda resolved: resolved["algorithm"].update({"provider": {"type": "custom_python", "distribution": None}}),
        ),
        (
            "unavailable",
            ComponentNotReconstructableError,
            lambda resolved: resolved["backend"]["kernel"].update({"component_id": "vamos.kernel:cuda@1"}),
        ),
    ],
)
def test_unsupported_components_and_providers_refuse_before_execution(
    expected_status: str,
    expected_error: type[ReplayUnavailableError],
    mutate: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = _save_source(tmp_path / "source")
    _rewrite_resolved(source.root, mutate)
    replay_module = importlib.import_module("vamos.experiment.artifacts.replay")
    monkeypatch.setattr(replay_module, "_run_config", lambda *args, **kwargs: pytest.fail("optimization executed"))

    report = vamos.verify_run(source.root)
    with pytest.raises(expected_error) as caught:
        vamos.reproduce(source.root, output=tmp_path / "replay")

    assert report.component_reconstructability == expected_status
    assert caught.value.optimization_executed is False
    assert not (tmp_path / "replay").exists()


def _zdt1_lookalike(x: np.ndarray) -> list[float]:
    # User code under a registry name: not the registered ZDT1.
    return [float(x[0]), float(1.0 + x[1] - x[0] ** 0.5)]


def test_make_problem_named_like_registry_is_not_exact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    problem = vamos.make_problem(_zdt1_lookalike, n_var=6, n_obj=2, bounds=[(0.0, 1.0)] * 6, encoding="real", name="zdt1")
    result = vamos.optimize(problem, algorithm="nsgaii", pop_size=8, max_evaluations=16, engine="numpy", seed=7)
    stored = vamos.save_result(result, tmp_path / "custom")
    replay_module = importlib.import_module("vamos.experiment.artifacts.replay")
    monkeypatch.setattr(replay_module, "_run_config", lambda *args, **kwargs: pytest.fail("optimization executed"))

    descriptor = stored.manifest.resolved_spec["problem"]
    assert descriptor["provider"]["type"] == "custom_python"
    assert descriptor["component_id"] != "vamos.problem:zdt1@1"
    assert descriptor["resolution"]["qualified_name"] == "_zdt1_lookalike"
    assert stored.manifest["replayability"]["declared_level"] == "manual"
    report = vamos.verify_run(stored.root)
    assert report.effective_replayability == "manual"
    with pytest.raises(UnsupportedReplayProviderError):
        vamos.reproduce(stored.root, output=tmp_path / "replay")
    assert not (tmp_path / "replay").exists()
    # The same dimensions, seed and configuration on the registered ZDT1 are a different task.
    assert _save_source(tmp_path / "registry", seed=7).manifest.task_id != stored.manifest.task_id


def test_registry_problem_instance_stays_exact(tmp_path: Path) -> None:
    # A WFG instance carries its registry key as its name; its class is the registered one.
    runs = {}
    for label, problem in (("instance", vamos.make_problem_selection("wfg4").instantiate()), ("name", "wfg4")):
        result = vamos.optimize(problem, algorithm="nsgaii", pop_size=8, max_evaluations=16, engine="numpy", seed=7)
        runs[label] = vamos.save_result(result, tmp_path / label)

    assert runs["instance"].manifest.resolved_spec["problem"]["provider"]["type"] == "built_in"
    assert runs["instance"].manifest.task_id == runs["name"].manifest.task_id
    assert vamos.verify_run(runs["instance"].root).effective_replayability == "exact"


def test_exact_replay_preserves_source_and_records_lineage(tmp_path: Path) -> None:
    source = _save_source(tmp_path / "source", seed=0)
    before = _snapshot(source.root)

    report = vamos.reproduce(source.root, output=tmp_path / "replay")

    assert isinstance(report, ReplayReport)
    assert report.exact is True
    assert report.source_run_id != report.replay_run_id
    assert report.task_id == source.manifest.task_id
    assert {"F", "X"}.issubset({item.role for item in report.comparisons if item.exact})
    assert {"population/F", "population/X"}.issubset({item.role for item in report.comparisons if item.exact})
    replay = vamos.load_run(report.output_root, verify="all")
    lineage = replay.manifest["lineage"]
    assert lineage["source_run_id"] == source.manifest.run_id
    assert lineage["root_run_id"] == source.manifest.run_id
    assert lineage["source_manifest_sha256"] == source.manifest["integrity"]["manifest_sha256"]
    assert lineage["comparison"]["status"] == "exact_match"
    assert _snapshot(source.root) == before


def test_seed_none_replays_persisted_concrete_seed(tmp_path: Path) -> None:
    source = _save_source(tmp_path / "source", seed=None)

    report = vamos.reproduce(source.root, output=tmp_path / "replay")
    replay = vamos.load_run(report.output_root)

    assert source.manifest.requested_spec["defaults"]["seed"] is None
    assert isinstance(source.manifest.resolved_spec["seed"], int)
    assert replay.manifest.resolved_spec["seed"] == source.manifest.resolved_spec["seed"]


def test_replay_does_not_call_current_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = (
        NSGAIIConfig.builder()
        .pop_size(8)
        .offspring_size(4)
        .crossover("sbx", prob=0.7, eta=11.0)
        .mutation("pm", prob=0.25, eta=31.0)
        .selection("tournament", size=3)
        .result_mode("population")
        .build()
    )
    source = _save_source(tmp_path / "source", config=config)
    monkeypatch.setattr(NSGAIIConfig, "default", classmethod(lambda cls, **kwargs: pytest.fail("current default used")))

    report = vamos.reproduce(source.root, output=tmp_path / "replay")

    assert report.exact


def test_replay_reconstructs_hv_stopping_and_external_archive(tmp_path: Path) -> None:
    config = (
        NSGAIIConfig.builder()
        .pop_size(8)
        .offspring_size(8)
        .crossover("sbx", prob=0.9, eta=15.0)
        .mutation("pm", prob=1.0 / 6.0, eta=20.0)
        .selection("tournament", size=2)
        .external_archive(capacity=12, pruning="crowding")
        .result_mode("population")
        .build()
    )
    termination = (
        "hv",
        {"target_value": 1e-4, "reference_point": [5.0, 5.0], "max_evaluations": 40},
    )
    result = vamos.optimize(
        "zdt1",
        algorithm="nsgaii",
        algorithm_config=config,
        termination=termination,
        engine="numpy",
        seed=3,
        n_var=6,
    )
    source = vamos.save_result(result, tmp_path / "source")

    replay = vamos.reproduce(source.root, output=tmp_path / "replay")

    assert replay.exact
    assert {"archive/F", "archive/X"}.issubset({item.role for item in replay.comparisons if item.exact})


def test_unknown_resolved_field_refuses_before_execution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _save_source(tmp_path / "source")
    _rewrite_resolved(source.root, lambda resolved: resolved["algorithm"]["config"].update({"future_default": 17}))
    replay_module = importlib.import_module("vamos.experiment.artifacts.replay")
    monkeypatch.setattr(replay_module, "_run_config", lambda *args, **kwargs: pytest.fail("optimization executed"))

    with pytest.raises(ResolvedSpecMismatchError) as caught:
        vamos.reproduce(source.root, output=tmp_path / "replay")
    assert caught.value.optimization_executed is False
    assert not (tmp_path / "replay").exists()


def test_replay_of_replay_retains_bounded_root_lineage(tmp_path: Path) -> None:
    source = _save_source(tmp_path / "source")
    first = vamos.reproduce(source.root, output=tmp_path / "first")

    second = vamos.reproduce(first.output_root, output=tmp_path / "second")
    lineage = vamos.load_run(second.output_root).manifest["lineage"]

    assert lineage["source_run_id"] == first.replay_run_id
    assert lineage["root_run_id"] == source.manifest.run_id
    assert lineage["depth"] == 2


def test_replay_refuses_to_exceed_lineage_depth_before_execution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _save_source(tmp_path / "source")
    first = vamos.reproduce(source.root, output=tmp_path / "first")
    manifest = _read_manifest(first.output_root)
    manifest["lineage"]["depth"] = 64
    _write_manifest(first.output_root, manifest)
    replay_module = importlib.import_module("vamos.experiment.artifacts.replay")
    monkeypatch.setattr(replay_module, "_run_config", lambda *args, **kwargs: pytest.fail("optimization executed"))

    with pytest.raises(ReplayUnavailableError) as caught:
        vamos.reproduce(first.output_root, output=tmp_path / "overflow")

    assert caught.value.optimization_executed is False
    assert not (tmp_path / "overflow").exists()


def test_default_output_is_discoverable_and_relocatable(tmp_path: Path) -> None:
    source = _save_source(tmp_path / "runs" / "source")

    report = vamos.reproduce(source.root)
    moved = tmp_path / "moved-replay"
    shutil.move(report.output_root, moved)

    assert report.output_root.parent == source.root.parent / "replays"
    assert vamos.load_run(moved, verify="all").manifest.run_id == report.replay_run_id


def test_output_collision_refuses_before_execution(tmp_path: Path) -> None:
    source = _save_source(tmp_path / "source")
    output = tmp_path / "occupied"
    output.mkdir()

    with pytest.raises(OutputCollisionError) as caught:
        vamos.reproduce(source.root, output=output)
    assert caught.value.optimization_executed is False


def test_execution_failure_creates_inspectable_failed_attempt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _save_source(tmp_path / "source")
    replay_module = importlib.import_module("vamos.experiment.artifacts.replay")
    monkeypatch.setattr(replay_module, "_run_config", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("safe failure")))
    output = tmp_path / "failed-replay"

    with pytest.raises(ReplayExecutionError) as caught:
        vamos.reproduce(source.root, output=output)

    assert caught.value.optimization_executed is True
    failed = vamos.load_run(output, verify="all")
    assert failed.status == "failed"
    assert failed.manifest["failure"]["phase"] == "optimization"
    assert failed.manifest["lineage"]["comparison"]["status"] == "execution_failed"
    assert (
        json.loads(json.dumps(importlib.import_module("vamos.experiment.artifacts.inspection").inspect_run(output)))["status"] == "failed"
    )


def test_exact_mismatch_is_stored_and_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = _save_source(tmp_path / "source")
    replay_module = importlib.import_module("vamos.experiment.artifacts.replay")
    original = replay_module._run_config

    def differing(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        changed = np.array(result.F, copy=True)
        changed.view(np.uint64).flat[0] ^= np.uint64(1)
        result.F = changed
        result.data["F"] = changed
        return result

    monkeypatch.setattr(replay_module, "_run_config", differing)
    output = tmp_path / "mismatch"

    with pytest.raises(ReplayResultMismatchError) as caught:
        vamos.reproduce(source.root, output=output)

    assert caught.value.optimization_executed is True
    mismatch = vamos.load_run(output, verify="all")
    assert mismatch.manifest["lineage"]["comparison"]["status"] == "mismatch"
    assert mismatch.manifest["replayability"]["declared_level"] == "unavailable"
    differences = [item for item in mismatch.manifest["lineage"]["comparison"]["arrays"] if not item["exact"]]
    assert differences[0]["first_difference"] == (0, 0)
    assert differences[0]["stored_sha256"] != differences[0]["replay_sha256"]


@pytest.mark.parametrize(
    ("stored", "replay", "classification"),
    [
        (np.array([0.0], dtype="<f8"), np.array([-0.0], dtype="<f8"), "content_mismatch"),
        (np.array([1.0], dtype="<f8"), np.array([1.0], dtype="<f4"), "dtype_mismatch"),
        (np.array([[1.0]], dtype="<f8"), np.array([1.0], dtype="<f8"), "shape_mismatch"),
        (np.array([1.0, 2.0]), np.array([2.0, 1.0]), "content_mismatch"),
    ],
)
def test_exact_comparison_distinguishes_layout_semantics(stored: np.ndarray, replay: np.ndarray, classification: str) -> None:
    comparisons = compare_array_collections({"F": stored, "X": stored}, {"F": replay, "X": replay})

    assert not comparisons_are_exact(comparisons)
    assert all(item.mismatch == classification for item in comparisons)
    assert all(item.stored_sha256 != item.replay_sha256 or classification != "content_mismatch" for item in comparisons)


def test_exact_comparison_preserves_nan_payload_bits() -> None:
    left = np.array([0x7FF8000000000001], dtype=np.uint64).view(np.float64)
    right = np.array([0x7FF8000000000002], dtype=np.uint64).view(np.float64)

    comparison = compare_array_collections({"F": left, "X": left}, {"F": right, "X": right})

    assert all(item.mismatch == "content_mismatch" for item in comparison)
    assert all(item.maximum_absolute_difference is None for item in comparison)


def test_exact_comparison_reports_missing_auxiliary_array_and_numeric_summary() -> None:
    stored = {"F": np.array([1.0, 2.0]), "X": np.array([1.0, 2.0]), "CV": np.array([0.0])}
    replay = {"F": np.array([1.0, 3.0]), "X": np.array([1.0, 2.0])}

    comparisons = {item.role: item for item in compare_array_collections(stored, replay)}

    assert comparisons["CV"].mismatch == "missing_replay_array"
    assert comparisons["F"].first_difference == (1,)
    assert comparisons["F"].maximum_absolute_difference == 1.0
