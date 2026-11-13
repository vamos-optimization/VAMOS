from __future__ import annotations

import dataclasses
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

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


def _generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_v1_snapshot_generator", ROOT / "tools" / "generate_v1_compatibility_snapshots.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _committed(name: str) -> object:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_cli_snapshot_detects_a_removed_option_that_is_a_substring(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--population-size` is a substring of `--offspring-population-size` (VR-091)."""
    generator = _generator()
    monkeypatch.setattr(generator, "STABLE_CLI", {"vamos": generator.STABLE_CLI["vamos"]})
    actual = generator._cli_arguments

    def without_population_size(command: str) -> frozenset[str]:
        return actual(command) - {"--population-size"}

    monkeypatch.setattr(generator, "_cli_arguments", without_population_size)
    with pytest.raises(RuntimeError, match="--population-size"):
        generator._stable_cli_snapshot()


def test_snapshot_detects_a_changed_config_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed default of a stable configuration changes the snapshot (VR-091)."""
    from vamos.algorithms import NSGAIIConfig

    generator = _generator()
    # algorithm_ids also lists plugins that other tests register in this process (VR-105).
    keys = ("config_fields", "config_field_defaults", "default_configs")
    committed = _committed("stable_algorithm_configs.json")
    assert {key: generator._algorithm_snapshot()[key] for key in keys} == {key: committed[key] for key in keys}
    default = NSGAIIConfig.default

    def changed_default(*args: object, **kwargs: object) -> NSGAIIConfig:
        return dataclasses.replace(default(*args, **kwargs), crossover=("sbx", {"prob": 1.0, "eta": 20.0}))

    monkeypatch.setattr(NSGAIIConfig, "default", changed_default)
    assert {key: generator._algorithm_snapshot()[key] for key in keys} != {key: committed[key] for key in keys}


def test_snapshot_detects_a_changed_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed default or parameter of a stable function changes the snapshot (VR-091)."""
    import vamos

    generator = _generator()
    assert generator._stable_api_snapshot() == _committed("stable_public_api.json")
    optimize = vamos.optimize

    def other_seed(problem: object, algorithm: str = "auto", *, seed: int = 7, **kwargs: object) -> object:
        return optimize(problem, algorithm, seed=seed, **kwargs)

    monkeypatch.setattr(vamos, "optimize", other_seed)
    assert generator._stable_api_snapshot() != _committed("stable_public_api.json")


def test_exit_code_snapshot_reads_the_cli_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """The frozen exit codes come from the code that maps errors to codes (VR-091)."""
    from vamos.experiment.artifacts.errors import OutputCollisionError
    from vamos.experiment.cli import run_artifact_cli

    generator = _generator()
    assert generator._exit_code_snapshot() == _committed("stable_exit_codes.json")
    actual = run_artifact_cli.run_artifact_exit_code
    monkeypatch.setattr(run_artifact_cli, "run_artifact_exit_code", lambda exc: 9 if isinstance(exc, OutputCollisionError) else actual(exc))
    assert generator._exit_code_snapshot()["run"]["output_collision"] == 9
