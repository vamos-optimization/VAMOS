"""Bounded loading and hash-chain validation for study journals."""

from __future__ import annotations

import re
from pathlib import Path
from typing import NoReturn

from vamos.experiment.artifacts.jsonio import sha256_bytes

from .errors import StudyCheckpointError, StudyIntegrityError
from .limits import StudyLoadLimits
from .models import StudyEvent, StudyManifest
from .paths import confined_study_path
from .record_decoding import decode_event
from .record_loading import ReadDocument
from .writing import is_writer_temporary

_EVENT_FILE_PATTERN = r"(\d{20})\.json"


def load_event_journal(
    root: Path,
    manifest: StudyManifest,
    limits: StudyLoadLimits,
    read: ReadDocument,
) -> tuple[StudyEvent, ...]:
    """Load the complete journal, including valid events newer than checkpoints."""
    directory = confined_study_path(root, "events", role="events", must_exist=True)
    try:
        # A writer temporary is in flight or orphaned by a dead writer, never an event.
        names = {entry.name for entry in directory.iterdir() if not is_writer_temporary(entry.name)}
    except OSError as exc:
        _integrity("UNREADABLE_DOCUMENT_DIRECTORY", "events", "readable event directory", type(exc).__name__, cause=exc)
    parsed: dict[int, str] = {}
    for name in names:
        match = re.fullmatch(_EVENT_FILE_PATTERN, name)
        if match is None:
            _integrity("EVENT_HASH_CHAIN_BROKEN", "events", "only 20-digit event JSON files", name)
        sequence = int(match.group(1))
        if sequence < 1 or sequence in parsed:
            _integrity("EVENT_HASH_CHAIN_BROKEN", "events", "unique positive event sequence", sequence)
        parsed[sequence] = name
    expected = set(range(1, len(parsed) + 1))
    if set(parsed) != expected:
        _integrity("EVENT_HASH_CHAIN_BROKEN", "events", sorted(expected), sorted(parsed))
    if manifest.checkpoint_sequence > len(parsed):
        raise StudyCheckpointError(
            operation="load study",
            reason="CHECKPOINT_AHEAD_OF_JOURNAL",
            document_role="study_manifest",
            field="$.checkpoint.sequence",
            expected=f"at most journal head {len(parsed)}",
            actual=manifest.checkpoint_sequence,
            action="Restore the missing immutable events; checkpoints never lead the journal.",
        )
    events: list[StudyEvent] = []
    event_ids: set[str] = set()
    previous: str | None = None
    for sequence in range(1, len(parsed) + 1):
        relative = f"events/{sequence:020d}.json"
        raw, payload = read(relative, "study_event", limits.max_event_bytes)
        event = decode_event(raw, file_sha256=sha256_bytes(payload))
        if event.sequence != sequence or event.previous_event_sha256 != previous:
            _integrity(
                "EVENT_HASH_CHAIN_BROKEN",
                relative,
                {"sequence": sequence, "previous_event_sha256": previous},
                {"sequence": event.sequence, "previous_event_sha256": event.previous_event_sha256},
            )
        if event.event_id in event_ids:
            _integrity("EVENT_HASH_CHAIN_BROKEN", relative, "globally unique event_id", event.event_id)
        events.append(event)
        event_ids.add(event.event_id)
        previous = event.file_sha256
    checkpoint = events[manifest.checkpoint_sequence - 1]
    if checkpoint.file_sha256 != manifest.checkpoint_event_sha256:
        raise StudyCheckpointError(
            operation="load study",
            reason="CHECKPOINT_JOURNAL_INCONSISTENCY",
            document_role="study_manifest",
            field="$.checkpoint.event_sha256",
            expected=checkpoint.file_sha256,
            actual=manifest.checkpoint_event_sha256,
            action="Restore the root checkpoint matching its referenced journal event.",
        )
    return tuple(events)


def _integrity(
    reason: str,
    role: str,
    expected: object,
    actual: object,
    *,
    cause: Exception | None = None,
) -> NoReturn:
    error = StudyIntegrityError(
        operation="load study",
        reason=reason,
        document_role=role,
        expected=expected,
        actual=actual,
        action="Restore the contiguous authoritative journal and matching checkpoints.",
    )
    if cause is None:
        raise error
    raise error from cause


__all__ = ["load_event_journal"]
