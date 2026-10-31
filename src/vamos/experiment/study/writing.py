"""Atomic file primitives shared by study creation and execution."""

from __future__ import annotations

import logging
import os
import re
import time
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar

from .serialization import stored_document_bytes

# Windows refuses to replace a file that another process holds open without delete
# sharing, which Python's open() never grants; readers hold a document for one read.
_RETRY_SHARING_VIOLATIONS = os.name == "nt"
_SHARING_ATTEMPTS = 20

_T = TypeVar("_T")


def write_document_atomic(path: Path, document: Mapping[str, Any]) -> bytes:
    """Atomically replace one mutable checkpoint with canonical bytes."""
    payload = stored_document_bytes(document)
    write_bytes_atomic(path, payload)
    return payload


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    """Publish one file atomically after flushing its bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".tmp-{uuid.uuid4().hex[:8]}")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            logging.getLogger(__name__).warning("Could not remove owned temporary file %s", temporary, exc_info=True)


def _replace(source: Path, target: Path) -> None:
    """Run ``os.replace``, retrying a Windows sharing violation.

    A reader in another process (``load_study``, ``inspect``, Studio) can hold ``target``
    open for one read; the wait outlasts that read.
    """
    _retry_sharing_violation(lambda: os.replace(source, target))


def read_bytes_shared(path: Path) -> bytes:
    """Read ``path`` in one call, retrying a Windows sharing violation.

    While the owning process replaces a document, Windows can refuse to open it for a
    moment; the wait outlasts that replace.
    """
    return _retry_sharing_violation(path.read_bytes)


def _retry_sharing_violation(operation: Callable[[], _T]) -> _T:
    """Run ``operation``, retrying a ``PermissionError`` on Windows for about 1.5 s.

    Elsewhere, and after the last attempt, the ``PermissionError`` propagates.
    """
    delay = 0.005
    for attempt in range(1, _SHARING_ATTEMPTS + 1):
        try:
            return operation()
        except PermissionError:
            if not _RETRY_SHARING_VIOLATIONS or attempt == _SHARING_ATTEMPTS:
                raise
            time.sleep(delay)
            delay = min(2.0 * delay, 0.1)
    raise AssertionError("unreachable: the last attempt returns or raises")


def is_writer_temporary(name: str) -> bool:
    """Return whether ``name`` has the form of a :func:`write_bytes_atomic` temporary.

    Readers skip these names: a temporary is either in flight or left by a writer that died
    before its replace, and never holds published state.
    """
    return re.fullmatch(r"\.tmp-[0-9a-f]{8}", name) is not None


def remove_writer_temporaries(directory: Path) -> None:
    """Delete orphaned writer temporaries directly inside ``directory``.

    Only reconciliation calls this, from the single process that owns the study, so no
    live writer can still be using them.
    """
    try:
        entries = list(directory.iterdir())
    except FileNotFoundError:
        return
    for entry in entries:
        if is_writer_temporary(entry.name) and entry.is_file() and not entry.is_symlink():
            entry.unlink(missing_ok=True)


def fsync_directory(path: Path) -> None:
    """Flush a directory entry where the platform exposes that primitive."""
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "fsync_directory",
    "is_writer_temporary",
    "read_bytes_shared",
    "remove_writer_temporaries",
    "write_bytes_atomic",
    "write_document_atomic",
]
