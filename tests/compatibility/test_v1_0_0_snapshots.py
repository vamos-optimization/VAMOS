from __future__ import annotations

import dataclasses
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

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


def _generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_v1_snapshot_generator", ROOT / "tools" / "generate_v1_compatibility_snapshots.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _committed(name: str) -> Any:
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
    committed = _committed("stable_algorithm_configs.json")

    def generated() -> dict[str, object]:
        return dict(json.loads(generator._json_bytes(generator._algorithm_snapshot())))

    added = generator._added_config_fields(committed, generated())
    assert added is not None
    assert generator._keeps_frozen_configs(committed, generated(), added)
    default = NSGAIIConfig.default

    def changed_default(*args: object, **kwargs: object) -> NSGAIIConfig:
        return dataclasses.replace(default(*args, **kwargs), crossover=("sbx", {"prob": 1.0, "eta": 20.0}))

    monkeypatch.setattr(NSGAIIConfig, "default", changed_default)
    assert not generator._keeps_frozen_configs(committed, generated(), added)


def test_snapshot_detects_a_changed_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed default or parameter of a stable function changes the snapshot (VR-091)."""
    import vamos

    generator = _generator()
    committed = _committed("stable_public_api.json")
    added = generator._added_config_fields(
        _committed("stable_algorithm_configs.json"), json.loads(generator._json_bytes(generator._algorithm_snapshot()))
    )
    assert added is not None

    def generated() -> object:
        return json.loads(generator._json_bytes(generator._stable_api_snapshot()))

    assert generator._keeps_frozen_api(committed, generated(), added)
    optimize = vamos.optimize

    def other_seed(problem: object, algorithm: str = "auto", *, seed: int = 7, **kwargs: object) -> object:
        return optimize(problem, algorithm, seed=seed, **kwargs)

    monkeypatch.setattr(vamos, "optimize", other_seed)
    assert not generator._keeps_frozen_api(committed, generated(), added)


def test_type_aliases_record_no_signature() -> None:
    """A type alias is not a callable API; whether typing makes one callable changes with the Python version."""
    import inspect

    generator = _generator()
    signatures = generator._stable_api_snapshot()["signatures"]
    aliases = [
        f"{module_name}.{name}"
        for module_name, names in generator.STABLE_API.items()
        for name in names
        if not (inspect.isroutine(obj := getattr(importlib.import_module(module_name), name)) or inspect.isclass(obj))
    ]

    assert "vamos.algorithms.ProbabilityValue" in aliases
    assert [alias for alias in aliases if alias in signatures] == []


def test_exit_code_snapshot_reads_the_cli_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """The frozen exit codes come from the code that maps errors to codes (VR-091)."""
    from vamos.experiment.artifacts.errors import OutputCollisionError
    from vamos.experiment.cli import run_artifact_cli

    generator = _generator()
    assert generator._exit_code_snapshot() == _committed("stable_exit_codes.json")
    actual = run_artifact_cli.run_artifact_exit_code
    monkeypatch.setattr(run_artifact_cli, "run_artifact_exit_code", lambda exc: 9 if isinstance(exc, OutputCollisionError) else actual(exc))
    assert generator._exit_code_snapshot()["run"]["output_collision"] == 9


def _configs(fields: list[str], defaults: dict[str, object], default_config: dict[str, object]) -> dict[str, Any]:
    return {
        "algorithm_ids": ["nsgaii"],
        "config_field_defaults": {"NSGAIIConfig": defaults},
        "config_fields": {"NSGAIIConfig": fields},
        "default_configs": {"NSGAIIConfig": default_config},
        "document_type": "vamos.stable-algorithm-configs",
    }


def test_configuration_classes_may_gain_optional_fields_after_the_frozen_ones() -> None:
    generator = _generator()
    frozen = _configs(
        ["pop_size", "constraint_mode"], {"constraint_mode": "feasibility"}, {"pop_size": 100, "constraint_mode": "feasibility"}
    )
    added = _configs(
        ["pop_size", "constraint_mode", "constraint_handling"],
        {"constraint_mode": "feasibility", "constraint_handling": None},
        {"pop_size": 100, "constraint_mode": "feasibility", "constraint_handling": None},
    )

    assert generator._added_config_fields(frozen, added) == {"NSGAIIConfig": ["constraint_handling"]}
    assert generator._keeps_frozen_configs(frozen, added, {"NSGAIIConfig": ["constraint_handling"]})
    # A new field before a frozen one would move the constructor's positional parameters.
    inserted = _configs(
        ["pop_size", "constraint_handling", "constraint_mode"], {"constraint_mode": "feasibility", "constraint_handling": None}, {}
    )
    assert generator._added_config_fields(frozen, inserted) is None
    assert generator._added_config_fields(frozen, _configs(["pop_size"], {}, {})) is None
    assert (
        generator._added_config_fields(
            frozen, _configs(["pop_size", "constraint_mode", "required"], {"constraint_mode": "feasibility"}, {})
        )
        is None
    )
    changed_default = _configs(
        ["pop_size", "constraint_mode"], {"constraint_mode": "none"}, {"pop_size": 100, "constraint_mode": "feasibility"}
    )
    assert not generator._keeps_frozen_configs(frozen, changed_default, {"NSGAIIConfig": []})
    changed_config = _configs(
        ["pop_size", "constraint_mode"], {"constraint_mode": "feasibility"}, {"pop_size": 50, "constraint_mode": "feasibility"}
    )
    assert not generator._keeps_frozen_configs(frozen, changed_config, {"NSGAIIConfig": []})


def test_only_the_constructor_of_a_configuration_class_with_added_fields_may_extend_its_signature() -> None:
    generator = _generator()
    frozen = "(pop_size, constraint_mode='feasibility')"
    added = {"NSGAIIConfig": ["constraint_handling"], "SPEA2Config": []}
    api: dict[str, Any] = {
        "modules": {"vamos.algorithms": ["NSGAIIConfig", "SPEA2Config"]},
        "signatures": {
            "vamos.algorithms.NSGAIIConfig": frozen,
            "vamos.algorithms.NSGAIIConfig.builder": "()",
            "vamos.algorithms.SPEA2Config": frozen,
        },
    }
    extended = "(pop_size, constraint_mode='feasibility', constraint_handling=None)"

    assert generator._extends_constructor(frozen, extended, ["constraint_handling"])
    assert not generator._extends_constructor(
        frozen, "(pop_size, constraint_handling=None, constraint_mode='feasibility')", ["constraint_handling"]
    )
    assert not generator._extends_constructor("(pop_size)", "(pop_size, constraint_handling)", ["constraint_handling"])
    assert not generator._extends_constructor(frozen, extended, [])

    def with_signature(key: str, signature: str) -> dict[str, Any]:
        return {**api, "signatures": {**api["signatures"], key: signature}}

    assert generator._keeps_frozen_api(api, with_signature("vamos.algorithms.NSGAIIConfig", extended), added)
    assert not generator._keeps_frozen_api(api, with_signature("vamos.algorithms.SPEA2Config", extended), added)
    assert not generator._keeps_frozen_api(api, with_signature("vamos.algorithms.NSGAIIConfig.builder", "(name)"), added)


def test_built_in_algorithms_may_be_added_but_not_removed() -> None:
    """Built-in algorithms added during 1.x (``ccmo``, ``ctaea``, ``top``) keep the 1.0 snapshot; a lost 1.0 identifier does not."""
    generator = _generator()
    frozen = _configs(["pop_size"], {}, {"pop_size": 100})
    added = {**frozen, "algorithm_ids": ["ccmo", "nsgaii"]}

    assert generator._keeps_frozen_configs(frozen, added, {"NSGAIIConfig": []})
    assert not generator._keeps_frozen_configs(frozen, {**frozen, "algorithm_ids": ["ccmo"]}, {"NSGAIIConfig": []})
    assert not generator._keeps_frozen_configs(frozen, {**frozen, "algorithm_ids": ["nsgaii", "nsgaii"]}, {"NSGAIIConfig": []})
    committed = _committed("stable_algorithm_configs.json")
    generated = json.loads(generator._json_bytes(generator._algorithm_snapshot()))
    for added_id in ("ccmo", "ctaea", "top"):
        assert added_id in generated["algorithm_ids"] and added_id not in committed["algorithm_ids"]
