from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = Path(__file__).resolve().parent / "v1_0_0"


def test_v1_0_0_structural_snapshots_are_current() -> None:
    result = subprocess.run(
        [sys.executable, "tools/generate_v1_compatibility_snapshots.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_v1_0_0_snapshot_documents_have_frozen_identity() -> None:
    for path in FIXTURES.glob("stable_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["document_type"].startswith("vamos.stable-")
        assert payload["schema_version"] == "1.0.0"


def test_v1_0_0_cli_snapshot_contains_only_supported_lifecycle_commands() -> None:
    payload = json.loads((FIXTURES / "stable_cli_tree.json").read_text(encoding="utf-8"))
    commands = set(payload["commands"])

    assert "vamos results inspect" in commands
    assert "vamos results verify" in commands
    assert "vamos reproduce" in commands
    assert "vamos study cancel" not in commands
    assert {"vamos study run", "vamos study resume", "vamos study retry"} <= commands


def _configs(**fields: list[str]) -> bytes:
    document = {"algorithm_ids": ["nsgaii"], "config_fields": fields, "document_type": "vamos.stable-algorithm-configs"}
    return json.dumps(document).encode("utf-8")


def test_configuration_classes_may_gain_fields_but_keep_the_frozen_ones() -> None:
    from tools.generate_v1_compatibility_snapshots import _keeps_frozen_fields

    frozen = _configs(NSGAIIConfig=["pop_size", "constraint_mode", "track_genealogy"]).decode("utf-8")

    assert _keeps_frozen_fields(frozen, _configs(NSGAIIConfig=["pop_size", "constraint_mode", "constraint_handling", "track_genealogy"]))
    assert not _keeps_frozen_fields(frozen, _configs(NSGAIIConfig=["pop_size", "track_genealogy"]))
    assert not _keeps_frozen_fields(frozen, _configs(NSGAIIConfig=["constraint_mode", "pop_size", "track_genealogy"]))
    assert not _keeps_frozen_fields(frozen, _configs(NSGAIIConfig=["pop_size", "constraint_mode", "track_genealogy"], SPEA2Config=[]))
