from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.experiment.artifacts import (
    ArtifactIntegrityError,
    ArtifactMissingError,
    ArtifactResourceLimitError,
    DuplicateJSONKeyError,
    MalformedResultBundleError,
    ManifestValidationError,
    OutputCollisionError,
    UnsafeArtifactPathError,
    UnsupportedArrayDTypeError,
    UnsupportedArtifactLayoutError,
    UnsupportedSchemaError,
)
from vamos.experiment.artifacts.bundle import inspect_result_bundle
from vamos.experiment.artifacts.jsonio import manifest_self_hash, sha256_file, stored_json_bytes
from vamos.experiment.artifacts.models import ArtifactDescriptor, LoadLimits
from vamos.experiment.optimization_result import OptimizationResult


def _result() -> OptimizationResult:
    source = json.loads(Path("docs/dev/run_artifact_examples/custom-manual/manifest.json").read_text(encoding="utf-8"))
    return OptimizationResult(
        {
            "F": np.array([[0.1, 0.9], [0.9, 0.1]], dtype=np.float64),
            "X": np.array([[1, 2], [3, 4]], dtype=np.int32),
            "evaluations": 2,
        },
        meta={
            "seed": 1,
            "run_artifact_requested_spec": source["requested_spec"],
            "run_artifact_resolved_spec": source["resolved_spec"],
        },
    )


def _descriptor(path: Path) -> ArtifactDescriptor:
    return ArtifactDescriptor(
        role="result_bundle",
        path=path.name,
        media_type="application/vnd.vamos.result-bundle+npz",
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        required_for=("load",),
        canonical=True,
    )


def test_missing_and_modified_result_are_actionable_ra017(tmp_path: Path) -> None:
    missing_run = vamos.save_result(_result(), tmp_path / "missing").root
    (missing_run / "result.npz").unlink()

    with pytest.raises(ArtifactMissingError) as missing_info:
        vamos.load_result(missing_run)
    assert missing_info.value.artifact_role == "result_bundle"
    assert missing_info.value.state == "missing"
    assert missing_info.value.expected_sha256

    corrupt_run = vamos.save_result(_result(), tmp_path / "corrupt").root
    result_path = corrupt_run / "result.npz"
    payload = bytearray(result_path.read_bytes())
    payload[-1] ^= 1
    result_path.write_bytes(payload)

    with pytest.raises(ArtifactIntegrityError) as corrupt_info:
        vamos.load_result(corrupt_run)
    assert corrupt_info.value.state == "hash_mismatch"
    assert corrupt_info.value.expected_sha256 != corrupt_info.value.actual_sha256

    length_run = vamos.save_result(_result(), tmp_path / "length").root
    with (length_run / "result.npz").open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(ArtifactIntegrityError) as length_info:
        vamos.load_result(length_run)
    assert length_info.value.state == "length_mismatch"
    assert length_info.value.actual_bytes == length_info.value.expected_bytes + 1


def test_manifest_semantic_edit_malformed_and_duplicate_keys_ra017(tmp_path: Path) -> None:
    semantic = vamos.save_result(_result(), tmp_path / "semantic").root / "manifest.json"
    semantic.write_text(semantic.read_text(encoding="utf-8").replace('"n_solutions": 2', '"n_solutions": 3'), encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError) as semantic_info:
        vamos.load_run(semantic.parent)
    assert semantic_info.value.artifact_role == "manifest"
    assert semantic_info.value.state == "hash_mismatch"

    malformed = vamos.save_result(_result(), tmp_path / "malformed").root / "manifest.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ManifestValidationError, match="malformed JSON"):
        vamos.load_run(malformed.parent)

    duplicate = vamos.save_result(_result(), tmp_path / "duplicate").root / "manifest.json"
    text = duplicate.read_text(encoding="utf-8")
    duplicate.write_text(text.replace('"status": "succeeded",', '"status": "succeeded",\n  "status": "failed",'), encoding="utf-8")
    with pytest.raises(DuplicateJSONKeyError):
        vamos.load_run(duplicate.parent)

    formatting = vamos.save_result(_result(), tmp_path / "formatting").root / "manifest.json"
    formatting.write_text(json.dumps(json.loads(formatting.read_text(encoding="utf-8"))), encoding="utf-8")
    assert vamos.load_run(formatting.parent).status == "succeeded"


def test_future_major_rejected_before_artifact_access_ra023(tmp_path: Path) -> None:
    sentinel = tmp_path / "sentinel"
    sentinel.write_text("untouched", encoding="utf-8")
    run = tmp_path / "future"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "document_type": "vamos.run-manifest",
                "schema_version": "2.0.0",
                "artifacts": [{"role": "result_bundle", "path": "../sentinel"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UnsupportedSchemaError):
        vamos.load_run(run)
    assert sentinel.read_text(encoding="utf-8") == "untouched"


@pytest.mark.parametrize("unsafe", ["../outside.npz", "/outside.npz", "C:/outside.npz", "server/share.npz%2fescape", "a\\b.npz"])
def test_unsafe_manifest_paths_are_rejected_ra018(unsafe: str, tmp_path: Path) -> None:
    run = vamos.save_result(_result(), tmp_path / "run").root
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    next(item for item in manifest["artifacts"] if item["role"] == "result_bundle")["path"] = unsafe
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(UnsafeArtifactPathError):
        vamos.load_run(run)


def test_symlink_escape_is_rejected_when_available_ra018(tmp_path: Path) -> None:
    run = vamos.save_result(_result(), tmp_path / "run").root
    outside = tmp_path / "outside"
    outside.mkdir()
    shutil_target = outside / "result.npz"
    shutil_target.write_bytes((run / "result.npz").read_bytes())
    link = run / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable in this environment: {exc}")
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    next(item for item in manifest["artifacts"] if item["role"] == "result_bundle")["path"] = "escape/result.npz"
    manifest["integrity"]["manifest_sha256"] = manifest_self_hash(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(UnsafeArtifactPathError):
        vamos.load_run(run)


def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    destination = tmp_path / "occupied"
    destination.mkdir()
    sentinel = destination / "sentinel.txt"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(OutputCollisionError):
        vamos.save_result(_result(), destination)
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert list(destination.iterdir()) == [sentinel]


@pytest.mark.parametrize("kind", ["empty", "valid", "partial"])
def test_all_existing_directory_states_collide(kind: str, tmp_path: Path) -> None:
    destination = tmp_path / "occupied"
    if kind == "valid":
        vamos.save_result(_result(), destination)
    else:
        destination.mkdir()
        if kind == "partial":
            (destination / "manifest.json").write_text('{"status": "running"}', encoding="utf-8")
    before = {item.name: item.read_bytes() for item in destination.iterdir()}

    with pytest.raises(OutputCollisionError):
        vamos.save_result(_result(), destination)
    assert {item.name: item.read_bytes() for item in destination.iterdir()} == before


@pytest.mark.parametrize("phase", ["bundle", "terminal_manifest", "publish"])
def test_injected_write_failure_publishes_nothing_and_cleans_staging_ra021(
    phase: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vamos.experiment.artifacts import storage

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise OSError(f"injected {phase} write failure")

    target = {
        "bundle": "write_result_bundle",
        "terminal_manifest": "build_terminal_manifest",
        "publish": "os.rename",
    }[phase]
    if "." in target:
        owner_name, attribute = target.split(".", 1)
        monkeypatch.setattr(getattr(storage, owner_name), attribute, fail)
    else:
        monkeypatch.setattr(storage, target, fail)
    destination = tmp_path / "failed-save"

    with pytest.raises(OSError, match="injected"):
        vamos.save_result(_result(), destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".*vamos-staging-*"))
    assert not list(tmp_path.glob(".*vamos-save.lock"))


def test_writer_snapshots_before_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vamos.experiment.artifacts import persistence

    result = _result()
    expected_f = result.F.copy()
    expected_x = result.X.copy()
    real_store = persistence.store_succeeded_run

    def mutate_then_store(*args: Any, **kwargs: Any) -> Any:
        result.F[:] = -100
        result.X[:] = -100
        return real_store(*args, **kwargs)

    monkeypatch.setattr(persistence, "store_succeeded_run", mutate_then_store)
    loaded = vamos.load_result(vamos.save_result(result, tmp_path / "snapshot").root)

    assert np.array_equal(loaded.F, expected_f)
    assert np.array_equal(loaded.X, expected_x)


def test_unsupported_dtype_is_rejected_without_pickle(tmp_path: Path) -> None:
    meta = _result().meta
    result = OptimizationResult(
        {"F": np.ones((1, 2)), "X": np.array([[{"unsafe": True}]], dtype=object)},
        meta=meta,
    )
    with pytest.raises(UnsupportedArrayDTypeError):
        vamos.save_result(result, tmp_path / "object")
    assert not (tmp_path / "object").exists()


@pytest.mark.parametrize(
    "dtype",
    [np.dtype("U2"), np.dtype("S2"), np.dtype("complex128"), np.dtype("datetime64[D]")],
)
def test_other_non_allowlisted_dtypes_are_rejected(dtype: np.dtype[object], tmp_path: Path) -> None:
    result = OptimizationResult({"F": np.ones((1, 2)), "X": np.zeros((1, 2), dtype=dtype)}, meta=_result().meta)
    with pytest.raises(UnsupportedArrayDTypeError):
        vamos.save_result(result, tmp_path / "unsupported")


@pytest.mark.parametrize("array", [np.array(1.0), np.ones(2), np.ones((1, 2, 3))])
def test_zero_dimensional_and_malformed_f_shapes_are_rejected(array: np.ndarray, tmp_path: Path) -> None:
    result = OptimizationResult({"F": array}, meta=_result().meta)
    with pytest.raises(MalformedResultBundleError):
        vamos.save_result(result, tmp_path / "bad-shape")


def test_object_npz_and_resource_limits_fail_before_materialization(tmp_path: Path) -> None:
    object_bundle = tmp_path / "object.npz"
    np.savez(object_bundle, F=np.array([[object()]], dtype=object))
    with pytest.raises(UnsupportedArrayDTypeError):
        inspect_result_bundle(
            object_bundle,
            descriptor=_descriptor(object_bundle),
            limits=LoadLimits(),
            required_f=True,
            operation="test bundle",
        )

    member_bundle = tmp_path / "members.npz"
    np.savez(member_bundle, F=np.ones((1, 2)), X=np.ones((1, 2)))
    with pytest.raises(ArtifactResourceLimitError) as member_info:
        inspect_result_bundle(
            member_bundle,
            descriptor=_descriptor(member_bundle),
            limits=LoadLimits(max_zip_members=1),
            required_f=True,
            operation="test bundle",
        )
    assert member_info.value.limit == "max_zip_members"

    compressed = tmp_path / "compressed.npz"
    np.savez_compressed(compressed, F=np.zeros((1000, 2)))
    with pytest.raises(ArtifactResourceLimitError) as ratio_info:
        inspect_result_bundle(
            compressed,
            descriptor=_descriptor(compressed),
            limits=LoadLimits(max_compression_ratio=2.0),
            required_f=True,
            operation="test bundle",
        )
    assert ratio_info.value.limit == "max_compression_ratio"


def test_malformed_and_duplicate_zip_members_are_rejected(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.npz"
    with zipfile.ZipFile(malformed, "w") as archive:
        archive.writestr("F.npy", b"not-an-npy")
    with pytest.raises(MalformedResultBundleError):
        inspect_result_bundle(
            malformed,
            descriptor=_descriptor(malformed),
            limits=LoadLimits(),
            required_f=True,
            operation="test bundle",
        )

    duplicate = tmp_path / "duplicate.npz"
    npy = tmp_path / "F.npy"
    with npy.open("wb") as handle:
        np.save(handle, np.ones((1, 2)))
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(duplicate, "w") as archive:
            archive.write(npy, "F.npy")
            archive.write(npy, "F.npy")
    with pytest.raises(MalformedResultBundleError, match="duplicate"):
        inspect_result_bundle(
            duplicate,
            descriptor=_descriptor(duplicate),
            limits=LoadLimits(),
            required_f=True,
            operation="test bundle",
        )


def test_verify_all_covers_environment(tmp_path: Path) -> None:
    run = vamos.save_result(_result(), tmp_path / "run").root
    (run / "environment.json").write_text("modified", encoding="utf-8")

    assert vamos.load_run(run, verify="required").status == "succeeded"
    with pytest.raises(ArtifactIntegrityError):
        vamos.load_run(run, verify="all")


def test_unknown_optional_role_is_preserved_but_never_opened(tmp_path: Path) -> None:
    run = vamos.save_result(_result(), tmp_path / "run").root
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"].append(
        {
            "role": "example.inert-extension",
            "path": "missing-extension.bin",
            "media_type": "application/octet-stream",
            "sha256": "0" * 64,
            "bytes": 999,
            "required_for": [],
            "canonical": False,
        }
    )
    manifest["example.extension"] = {"module": "must.not.be.imported"}
    manifest["integrity"] = {}
    manifest["integrity"] = {"manifest_sha256": manifest_self_hash(manifest)}
    manifest_path.write_bytes(stored_json_bytes(manifest))

    loaded = vamos.load_run(run, verify="all")

    assert loaded.manifest["example.extension"]["module"] == "must.not.be.imported"
    assert loaded.manifest.artifact("example.inert-extension") is not None
    assert not (run / "missing-extension.bin").exists()


def test_unknown_load_required_role_is_rejected_without_opening(tmp_path: Path) -> None:
    run = vamos.save_result(_result(), tmp_path / "run").root
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"].append(
        {
            "role": "example.future-canonical",
            "path": "missing-future.bin",
            "media_type": "application/octet-stream",
            "sha256": "0" * 64,
            "bytes": 99,
            "required_for": ["load"],
            "canonical": True,
        }
    )
    manifest["integrity"] = {}
    manifest["integrity"] = {"manifest_sha256": manifest_self_hash(manifest)}
    manifest_path.write_bytes(stored_json_bytes(manifest))

    with pytest.raises(UnsupportedSchemaError, match="unknown artifact role required for loading"):
        vamos.load_run(run)
    assert not (run / "missing-future.bin").exists()


def test_noncanonical_layout_is_actionably_rejected_without_mutation(tmp_path: Path) -> None:
    run = tmp_path / "unsupported"
    run.mkdir()
    (run / "unknown.bin").write_bytes(b"not a run")
    before = {item.name: item.read_bytes() for item in run.iterdir()}

    with pytest.raises(UnsupportedArtifactLayoutError, match="pre-1.0 development format; regenerate"):
        vamos.load_run(run)
    assert {item.name: item.read_bytes() for item in run.iterdir()} == before


def test_stale_save_lock_error_names_the_lock(tmp_path: Path) -> None:
    destination = tmp_path / "locked"
    lock = tmp_path / ".locked.vamos-save.lock"
    holder = {"token": "0" * 32, "pid": 999999, "host": "elsewhere", "created_at": "2026-01-01T00:00:00Z"}
    lock.write_text(json.dumps(holder), encoding="utf-8")
    orphan = tmp_path / f".locked.vamos-staging-{'1' * 32}"
    orphan.mkdir()

    with pytest.raises(OutputCollisionError) as caught:
        vamos.save_result(_result(), destination)

    error = caught.value
    assert str(lock) in error.action
    assert str(orphan) in error.action
    assert "999999" in error.reason and "elsewhere" in error.reason
    assert lock.exists() and orphan.exists()
    assert not destination.exists()


def test_save_lock_from_an_earlier_version_is_named_too(tmp_path: Path) -> None:
    lock = tmp_path / ".locked.vamos-save.lock"
    lock.write_bytes(b"0" * 32)

    with pytest.raises(OutputCollisionError) as caught:
        vamos.save_result(_result(), tmp_path / "locked")

    assert str(lock) in caught.value.action
    assert lock.read_bytes() == b"0" * 32


def test_save_lock_records_its_holder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os
    import socket

    from vamos.experiment.artifacts import storage

    seen: dict[str, Any] = {}
    real_write = storage.write_result_bundle

    def read_lock_then_write(*args: Any, **kwargs: Any) -> Any:
        seen.update(json.loads((tmp_path / ".held.vamos-save.lock").read_text(encoding="utf-8")))
        return real_write(*args, **kwargs)

    monkeypatch.setattr(storage, "write_result_bundle", read_lock_then_write)
    vamos.save_result(_result(), tmp_path / "held")

    assert seen["pid"] == os.getpid()
    assert seen["host"] == socket.gethostname()
    assert seen["created_at"].endswith("Z")
    assert not (tmp_path / ".held.vamos-save.lock").exists()


def _lock_holder_json() -> str:
    return json.dumps({"token": "0" * 32, "pid": 4242, "host": "elsewhere", "created_at": "2026-01-01T00:00:00Z"})


def test_existing_lock_is_read_only_as_a_small_regular_file(tmp_path: Path) -> None:
    import os

    lock = tmp_path / ".locked.vamos-save.lock"
    lock.write_text(_lock_holder_json() + " " * 100_000, encoding="utf-8")
    with pytest.raises(OutputCollisionError) as caught:
        vamos.save_result(_result(), tmp_path / "locked")
    assert str(lock) in caught.value.action and "4242" not in caught.value.reason

    lock.unlink()
    lock.mkdir()
    with pytest.raises(OutputCollisionError) as caught:
        vamos.save_result(_result(), tmp_path / "locked")
    assert str(lock) in caught.value.action

    lock.rmdir()
    target = tmp_path / "holder.json"
    target.write_text(_lock_holder_json(), encoding="utf-8")
    try:
        os.symlink(target, lock)
    except (OSError, NotImplementedError):
        return  # symbolic links need privileges on this platform
    with pytest.raises(OutputCollisionError) as caught:
        vamos.save_result(_result(), tmp_path / "locked")
    assert str(lock) in caught.value.action and "4242" not in caught.value.reason


@pytest.mark.skipif(not hasattr(__import__("os"), "mkfifo"), reason="FIFOs are POSIX only")
def test_fifo_at_the_lock_path_does_not_block(tmp_path: Path) -> None:
    import os

    lock = tmp_path / ".locked.vamos-save.lock"
    os.mkfifo(lock)

    with pytest.raises(OutputCollisionError) as caught:
        vamos.save_result(_result(), tmp_path / "locked")

    assert str(lock) in caught.value.action
