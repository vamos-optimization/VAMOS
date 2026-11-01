"""Non-destructive atomic directory storage for v1 run artifacts."""

from __future__ import annotations

import glob
import json
import logging
import os
import secrets
import shutil
import socket
import stat
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .bundle import array_contract, load_result_bundle, write_result_bundle
from .errors import ArtifactResourceLimitError, OutputCollisionError
from .jsonio import sha256_file, stored_json_bytes
from .manifest import build_terminal_manifest
from .models import ArtifactDescriptor, LoadLimits, RunManifest
from .paths import windows_path_length_error


def store_succeeded_run(
    destination: Path,
    *,
    arrays: Mapping[str, np.ndarray[Any, Any]],
    environment: Mapping[str, Any],
    manifest_base: Mapping[str, Any],
    limits: LoadLimits,
) -> RunManifest:
    """Write a complete run in a sibling staging directory and publish once."""
    return _store_terminal_run(
        destination,
        arrays=arrays,
        environment=environment,
        manifest_base=manifest_base,
        limits=limits,
    )


def store_failed_run(
    destination: Path,
    *,
    environment: Mapping[str, Any],
    manifest_base: Mapping[str, Any],
    limits: LoadLimits,
) -> RunManifest:
    """Write an inspectable failed attempt through the canonical publisher."""
    return _store_terminal_run(
        destination,
        arrays=None,
        environment=environment,
        manifest_base=manifest_base,
        limits=limits,
    )


def _store_terminal_run(
    destination: Path,
    *,
    arrays: Mapping[str, np.ndarray[Any, Any]] | None,
    environment: Mapping[str, Any],
    manifest_base: Mapping[str, Any],
    limits: LoadLimits,
) -> RunManifest:
    destination = destination.absolute()
    parent = destination.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        explained = windows_path_length_error(exc)
        if explained is None:
            raise
        raise explained from exc
    _reject_existing(destination)
    # 128 random bits in 22 characters rather than 32 hex digits, to keep staging paths short on Windows.
    token = secrets.token_urlsafe(16)
    staging = parent / f".{destination.name}.vamos-staging-{token}"
    lock = parent / f".{destination.name}.vamos-save.lock"
    acquired_lock = False
    owns_staging = False
    published = False
    try:
        try:
            with lock.open("xb") as handle:
                handle.write(_lock_record(token))
                handle.flush()
                os.fsync(handle.fileno())
            acquired_lock = True
        except (FileExistsError, PermissionError) as exc:
            # Windows refuses exclusive creation over an existing directory with PermissionError.
            if isinstance(exc, PermissionError) and not os.path.lexists(lock):
                raise
            raise _lock_collision(destination, lock) from exc
        _reject_existing(destination)
        staging.mkdir(exist_ok=False)
        owns_staging = True
        _write_running_manifest(staging, manifest_base)

        environment_path = staging / "environment.json"
        _write_bytes_atomic(environment_path, stored_json_bytes(environment))
        _check_stored_size(
            environment_path,
            limit=limits.max_environment_bytes,
            limit_name="max_environment_bytes",
            role="environment",
        )
        environment_descriptor = _descriptor(
            "environment",
            environment_path,
            "application/vnd.vamos.environment+json",
            required_for=["inspect", "verify", "replay"],
            canonical=True,
        )
        descriptors = [environment_descriptor]
        if arrays is not None:
            descriptors.append(_store_result(staging, arrays, limits))
        terminal_value = dict(manifest_base)
        terminal_value["artifacts"] = descriptors
        terminal = build_terminal_manifest(terminal_value, limits=limits)
        _write_bytes_atomic(staging / "manifest.json", stored_json_bytes(terminal.as_dict()))
        _fsync_directory(staging)
        _reject_existing(destination)
        os.rename(staging, destination)
        published = True
        _fsync_directory(parent)
        return terminal
    except FileExistsError as exc:
        raise _collision(destination, "already exists") from exc
    except OSError as exc:
        explained = windows_path_length_error(exc)
        if explained is None:
            raise
        raise explained from exc
    finally:
        if not published and owns_staging:
            _remove_owned_staging(staging, parent=parent, token=token)
        if acquired_lock:
            try:
                lock.unlink(missing_ok=True)
            except OSError:
                logging.getLogger(__name__).warning("Could not remove owned save lock %s", lock, exc_info=True)


def _store_result(staging: Path, arrays: Mapping[str, np.ndarray[Any, Any]], limits: LoadLimits) -> dict[str, Any]:
    result_path = staging / "result.npz"
    result_temp = staging / ".result.npz.tmp"
    write_result_bundle(result_temp, arrays, limits=limits)
    os.replace(result_temp, result_path)
    _check_stored_size(result_path, limit=limits.max_artifact_bytes, limit_name="max_artifact_bytes", role="result_bundle")
    result_descriptor = {
        **_descriptor(
            "result_bundle",
            result_path,
            "application/vnd.vamos.result-bundle+npz",
            required_for=["load", "inspect", "verify", "replay", "analysis"],
            canonical=True,
        ),
        "array_contract": array_contract(arrays),
    }
    load_result_bundle(
        result_path,
        descriptor=_artifact_descriptor(result_descriptor),
        limits=limits,
        required_f=True,
        operation="save result",
    )
    return result_descriptor


def _write_running_manifest(staging: Path, manifest_base: Mapping[str, Any]) -> None:
    running = dict(manifest_base)
    running["status"] = "running"
    running["artifacts"] = []
    running.pop("outcome", None)
    running.pop("integrity", None)
    timestamps = dict(running.get("timestamps", {}))
    timestamps.pop("completed_at", None)
    running["timestamps"] = timestamps
    _write_bytes_atomic(staging / "manifest.json", stored_json_bytes(running))


def _descriptor(
    role: str,
    path: Path,
    media_type: str,
    *,
    required_for: list[str],
    canonical: bool,
) -> dict[str, Any]:
    return {
        "role": role,
        "path": path.name,
        "media_type": media_type,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "required_for": required_for,
        "canonical": canonical,
    }


def _artifact_descriptor(value: Mapping[str, Any]) -> ArtifactDescriptor:
    contract = value.get("array_contract")
    return ArtifactDescriptor(
        role=str(value["role"]),
        path=str(value["path"]),
        media_type=str(value["media_type"]),
        sha256=str(value["sha256"]),
        bytes=int(value["bytes"]),
        required_for=tuple(str(item) for item in value["required_for"]),
        canonical=bool(value["canonical"]),
        array_contract=contract if isinstance(contract, Mapping) else None,
    )


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    temp = path.with_name(f".{path.name}.tmp")
    with temp.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _check_stored_size(path: Path, *, limit: int, limit_name: str, role: str) -> None:
    observed = path.stat().st_size
    if observed > limit:
        raise ArtifactResourceLimitError(
            operation="save result",
            limit=limit_name,
            configured=limit,
            observed=observed,
            artifact_role=role,
            path=path.name,
            action="Pass explicit trusted LoadLimits only when this output size is intended.",
        )


def _reject_existing(path: Path) -> None:
    if os.path.lexists(path):
        raise _collision(path, "already exists")


def _collision(path: Path, reason: str) -> OutputCollisionError:
    return OutputCollisionError(
        operation="save result",
        path=path,
        reason=reason,
        expected="a destination path that does not exist",
        actual="occupied output path",
        action="Choose another output directory; VAMOS never overwrites or merges run artifacts.",
    )


def _lock_record(token: str) -> bytes:
    """Who holds a save lock: the token, process, host and UTC start time."""
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    record = {"token": token, "pid": os.getpid(), "host": socket.gethostname(), "created_at": created_at}
    return json.dumps(record, sort_keys=True).encode("utf-8")


# A lock VAMOS writes is a few hundred bytes of JSON.
_LOCK_READ_LIMIT = 4096


def _read_lock_holder(lock: Path) -> object:
    """The JSON of an existing lock, read only from a small regular file; ``None`` otherwise.

    The path is untrusted: a link, a FIFO or a large file is never followed, opened or read whole.
    """
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    try:
        info = os.lstat(lock)
        if not stat.S_ISREG(info.st_mode) or info.st_size > _LOCK_READ_LIMIT:
            return None
        descriptor = os.open(lock, flags)
        try:
            data = os.read(descriptor, _LOCK_READ_LIMIT + 1)
        finally:
            os.close(descriptor)
        return json.loads(data.decode("utf-8")) if len(data) <= _LOCK_READ_LIMIT else None
    except (OSError, ValueError):
        return None


def _lock_collision(destination: Path, lock: Path) -> OutputCollisionError:
    """The destination's save lock exists: name it, its holder and any unfinished staging directories.

    A process that dies while saving leaves its lock and staging behind; VAMOS never removes them itself.
    """
    holder = _read_lock_holder(lock)
    reason = "is locked by another save"
    if isinstance(holder, dict) and {"pid", "host", "created_at"} <= holder.keys():
        reason += f" (process {holder['pid']} on host {holder['host']}, since {holder['created_at']})"
    action = f"Wait for that save to finish. If its process is no longer running, delete {lock} and save again"
    staging = sorted(destination.parent.glob(f".{glob.escape(destination.name)}.vamos-staging-*"))
    if staging:
        action += f"; its unfinished staging directories can be deleted too: {', '.join(str(path) for path in staging)}"
    return OutputCollisionError(
        operation="save result",
        path=destination,
        reason=reason,
        expected="no other save of the destination in progress",
        actual=f"save lock {lock}",
        action=f"{action}. VAMOS never removes save locks automatically.",
    )


def _remove_owned_staging(staging: Path, *, parent: Path, token: str) -> None:
    if not os.path.lexists(staging):
        return
    expected_name = f".vamos-staging-{token}"
    try:
        resolved_parent = parent.resolve(strict=True)
        resolved_staging_parent = staging.parent.resolve(strict=True)
    except OSError:
        return
    if resolved_staging_parent != resolved_parent or not staging.name.endswith(expected_name):
        return
    shutil.rmtree(staging)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["store_failed_run", "store_succeeded_run"]
