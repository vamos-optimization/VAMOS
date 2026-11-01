from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import vamos
from vamos.experiment.cli.study_command import StudyCommandRequest, execute_study_command, map_exit_code
from vamos.experiment.optimization_result import OptimizationResult
from vamos.experiment.study.errors import (
    InvalidStudySpecError,
    MalformedStudyError,
    StudyInfrastructureError,
    StudyOutputCollisionError,
    UnsupportedStudyExecutionStateError,
)


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    return subprocess.run(
        [sys.executable, "-m", "vamos.experiment.cli.main", "study", *args],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )


def _write_spec(path: Path, *, seeds: list[int] | None = None, empty: bool = False) -> None:
    value: dict[str, object] = {
        "problems": [] if empty else ["zdt1"],
        "algorithms": [] if empty else ["nsgaii"],
        "seeds": [] if empty else (seeds or [0]),
        "max_evaluations": 8,
        "pop_size": 4,
    }
    path.write_text(json.dumps(value), encoding="utf-8")


def _tree(root: Path) -> dict[str, bytes | None]:
    return {item.relative_to(root).as_posix(): item.read_bytes() if item.is_file() else None for item in sorted(root.rglob("*"))}


def _result(reconstructed: Any) -> OptimizationResult:
    return OptimizationResult(
        {
            "F": np.full((1, reconstructed.n_obj), reconstructed.seed, dtype=np.float64),
            "X": np.full((1, reconstructed.n_var), reconstructed.seed, dtype=np.float64),
            "evaluations": 8,
            "generations": 2,
            "metrics": {"score": float(reconstructed.seed)},
        }
    )


@pytest.mark.parametrize("command", ["plan", "create", "run", "inspect", "resume", "retry", "summarize"])
def test_every_study_command_has_side_effect_free_help(tmp_path: Path, command: str) -> None:
    before = set(tmp_path.iterdir())
    proc = _run(command, "--help")
    assert proc.returncode == 0
    assert f"vamos study {command}" in proc.stdout
    assert "--json" in proc.stdout
    assert set(tmp_path.iterdir()) == before


def test_sa_069_plan_create_and_inspect_share_identity_and_one_envelope(tmp_path: Path) -> None:
    config = tmp_path / "study.json"
    root = tmp_path / "study"
    _write_spec(config)

    planned = _run("plan", str(config), "--output", str(root), "--json")
    created = _run("create", str(config), "--output", str(root), "--json")
    before = _tree(root)
    inspected = _run("inspect", str(root), "--json")

    assert planned.returncode == created.returncode == inspected.returncode == 0
    envelopes = [json.loads(item.stdout) for item in (planned, created, inspected)]
    assert {item["document_type"] for item in envelopes} == {"vamos.study-command-result"}
    assert {item["schema_version"] for item in envelopes} == {"1.0.0"}
    assert envelopes[0]["plan_id"] == envelopes[1]["plan_id"] == envelopes[2]["plan_id"]
    assert envelopes[1]["study_id"] == envelopes[2]["study_id"]
    assert envelopes[1]["changed"] is True
    assert envelopes[1]["payload"]["execution_began"] is False
    assert envelopes[2]["changed"] is False
    assert envelopes[2]["payload"]["report"] == vamos.load_study(root).inspect().as_dict()
    assert envelopes[0]["payload"]["output"]["requested_path"] == root.name
    assert str(root) not in planned.stdout
    assert _tree(root) == before
    assert all(item.stdout.count("\n") == 1 for item in (planned, created, inspected))
    assert "Concurrent mutation is unsupported" in created.stderr
    assert planned.stderr.count("Warning:") == 2
    assert inspected.stderr == ""


def test_human_create_and_inspect_disclose_single_owner_without_traceback(tmp_path: Path) -> None:
    config = tmp_path / "study.json"
    root = tmp_path / "study"
    _write_spec(config)

    created = _run("create", str(config), "--output", str(root))
    inspected = _run("inspect", str(root))

    assert created.returncode == inspected.returncode == 0
    assert "Study create: created" in created.stdout
    assert "Canonical state changed: yes" in created.stdout
    assert "Concurrent mutation is unsupported" in created.stderr
    assert "Study inspect: created" in inspected.stdout
    assert "Traceback" not in created.stderr + inspected.stderr


def test_successful_run_and_read_only_summary_human_and_json(tmp_path: Path) -> None:
    config = tmp_path / "study.json"
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_spec(config)
    assert _run("create", str(config), "--output", str(first), "--json").returncode == 0
    assert _run("create", str(config), "--output", str(second), "--json").returncode == 0

    human = _run("run", str(first))
    machine = _run("run", str(second), "--json")
    before = _tree(second)
    summary = _run("summarize", str(second), "--json")

    assert human.returncode == machine.returncode == summary.returncode == 0
    assert "Study run: completed" in human.stdout
    assert "Execution began: yes" in human.stdout
    run_result = json.loads(machine.stdout)
    assert run_result["status"] == "completed"
    assert run_result["changed"] is True
    assert run_result["payload"]["execution_began"] is True
    summary_result = json.loads(summary.stdout)
    assert summary_result["operation"] == "summarize"
    assert summary_result["changed"] is False
    assert len(summary_result["payload"]["summary"]["rows"]) == 1
    assert summary_result["payload"]["summary"] == vamos.load_study(second).summarize().as_dict()
    assert _tree(second) == before


def test_summary_json_and_csv_are_explicit_atomic_traceable_and_collision_safe(tmp_path: Path) -> None:
    config = tmp_path / "study.json"
    root = tmp_path / "study"
    json_output = tmp_path / "summary.json"
    csv_output = tmp_path / "summary.csv"
    _write_spec(config)
    assert _run("create", str(config), "--output", str(root), "--json").returncode == 0
    canonical_before = _tree(root)

    json_proc = _run("summarize", str(root), "--format", "json", "--output", str(json_output), "--json")
    csv_proc = _run("summarize", str(root), "--format", "csv", "--output", str(csv_output), "--json")
    collision = _run("summarize", str(root), "--format", "json", "--output", str(json_output), "--json")

    assert json_proc.returncode == csv_proc.returncode == 0
    assert collision.returncode == 5
    summary = json.loads(json_output.read_text(encoding="utf-8"))
    assert summary["document_type"] == "vamos.study-summary"
    assert summary["derived"] is True
    assert summary["canonical_authority"] is False
    assert summary["study_id"] == vamos.load_study(root).study_id
    with csv_output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["study_id"] == summary["study_id"]
    assert rows[0]["plan_id"] == summary["plan_id"]
    assert rows[0]["event_head_sha256"] == summary["event_head"]["sha256"]
    assert rows[0]["derived"] == "True"
    assert rows[0]["canonical_authority"] == "False"
    assert json.loads(collision.stdout)["errors"][0]["reason"] == "OUTPUT_COLLISION"
    assert _tree(root) == canonical_before

    first_bytes = json_output.read_bytes()
    json_output.unlink()
    regenerated = _run("summarize", str(root), "--format", "json", "--output", str(json_output), "--json")
    assert regenerated.returncode == 0
    assert json_output.read_bytes() == first_bytes

    unsafe = _run("summarize", str(root), "--output", str(root / "not-derived.json"), "--json")
    assert unsafe.returncode == 3
    assert json.loads(unsafe.stdout)["errors"][0]["reason"] == "UNSAFE_SUMMARY_DESTINATION"
    assert not (root / "not-derived.json").exists()


def test_empty_run_resume_and_retry_use_current_services(tmp_path: Path) -> None:
    config = tmp_path / "study.json"
    root = tmp_path / "study"
    _write_spec(config, empty=True)
    assert _run("create", str(config), "--output", str(root), "--json").returncode == 0

    completed = _run("run", str(root), "--json")
    resumed = _run("resume", str(root), "--json")
    retried = _run("retry", str(root), "--failed", "--json")

    assert completed.returncode == resumed.returncode == retried.returncode == 0
    assert json.loads(completed.stdout)["status"] == "completed"
    assert json.loads(resumed.stdout)["changed"] is False
    assert json.loads(retried.stdout)["changed"] is False


def test_cli_run_ctrl_c_uses_durable_graceful_cancellation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    root = tmp_path / "study"
    vamos.create_study(
        vamos.StudySpec(problems=["zdt1"], algorithms=["nsgaii"], seeds=[0], max_evaluations=8, pop_size=4),
        output=root,
    )

    def interrupt(_reconstructed: Any, *, root: Path) -> OptimizationResult:
        raise KeyboardInterrupt

    monkeypatch.setattr(execution, "_execute_optimization", interrupt)
    result = execute_study_command(StudyCommandRequest(operation="run", study_dir=root, json_output=True))

    assert result.exit_code == 8
    assert result.status == "cancelled"
    assert result.changed is True
    assert result.payload["execution_began"] is True
    assert vamos.load_study(root).status == "cancelled"


def test_sa_067_cli_inspect_and_summarize_do_not_materialize_arrays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.artifacts.reader as reader
    import vamos.experiment.study.execution as execution

    root = tmp_path / "study"
    monkeypatch.setattr(execution, "_execute_optimization", lambda reconstructed, *, root: _result(reconstructed))
    vamos.create_study(
        vamos.StudySpec(problems=["zdt1"], algorithms=["nsgaii"], seeds=[0], max_evaluations=8, pop_size=4),
        output=root,
    ).run()

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("CLI projection must not materialize F or X")

    monkeypatch.setattr(reader, "load_result_bundle", forbidden)
    before = _tree(root)
    inspected = execute_study_command(StudyCommandRequest(operation="inspect", study_dir=root, json_output=True))
    summarized = execute_study_command(StudyCommandRequest(operation="summarize", study_dir=root, json_output=True))

    assert inspected.exit_code == summarized.exit_code == 0
    assert inspected.changed is summarized.changed is False
    assert _tree(root) == before


def test_corrupt_referenced_run_is_reported_with_exit_three_without_repair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    root = tmp_path / "study"
    monkeypatch.setattr(execution, "_execute_optimization", lambda reconstructed, *, root: _result(reconstructed))
    completed = vamos.create_study(
        vamos.StudySpec(problems=["zdt1"], algorithms=["nsgaii"], seeds=[0], max_evaluations=8, pop_size=4),
        output=root,
    ).run()
    reference = completed.attempts[0].run_reference
    assert reference is not None
    (root / str(reference["path"])).unlink()
    before = _tree(root)

    inspected = execute_study_command(StudyCommandRequest(operation="inspect", study_dir=root, json_output=True))

    assert inspected.exit_code == 3
    assert inspected.changed is False
    assert inspected.errors[0]["reason"] == "REFERENCED_RUN_MISSING"
    assert _tree(root) == before


def test_fail_fast_partial_maps_to_six_and_explicit_retry_uses_new_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.study.execution as execution

    root = tmp_path / "study"
    vamos.create_study(
        vamos.StudySpec(problems=["zdt1"], algorithms=["nsgaii"], seeds=[0], max_evaluations=8, pop_size=4),
        output=root,
    )
    monkeypatch.setattr(execution, "_execute_optimization", lambda _reconstructed, *, root: (_ for _ in ()).throw(RuntimeError("task")))
    failed = execute_study_command(StudyCommandRequest(operation="run", study_dir=root))
    inspected = execute_study_command(StudyCommandRequest(operation="inspect", study_dir=root))
    first_attempt = vamos.load_study(root).attempts[0].attempt_id
    monkeypatch.setattr(execution, "_execute_optimization", lambda reconstructed, *, root: _result(reconstructed))
    retried = execute_study_command(StudyCommandRequest(operation="retry", study_dir=root, failed_only=True))

    assert failed.exit_code == 6
    assert failed.status == "paused"
    assert inspected.exit_code == 6
    assert inspected.status == "paused"
    assert retried.exit_code == 0
    assert retried.status == "completed"
    attempts = vamos.load_study(root).attempts
    assert len(attempts) == 2
    assert attempts[0].attempt_id == first_attempt
    assert attempts[1].attempt_id != first_attempt


def test_sa_072_exit_code_table_is_stable() -> None:
    common = {
        "operation": "test study",
        "expected": "expected",
        "actual": "actual",
        "action": "inspect",
    }
    assert map_exit_code("inspect", "completed_with_failures") == 6
    assert map_exit_code("run", InvalidStudySpecError(reason="INVALID_STUDY_SPEC", **common)) == 2
    assert map_exit_code("run", MalformedStudyError(reason="MALFORMED_JSON", **common)) == 3
    assert map_exit_code("run", UnsupportedStudyExecutionStateError(reason="INVALID_STATE_TRANSITION", **common)) == 4
    assert map_exit_code("run", StudyOutputCollisionError(reason="OUTPUT_COLLISION", **common)) == 5
    assert map_exit_code("run", StudyInfrastructureError(reason="INFRASTRUCTURE", **common)) == 7
    assert map_exit_code("run", "cancelled") == 8


def test_unexpected_cli_failure_is_sanitized_without_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    import vamos.experiment.cli.study_command as command

    def fail(_request: StudyCommandRequest) -> Any:
        raise RuntimeError("private secret traceback")

    monkeypatch.setitem(command._COMMANDS, "inspect", fail)
    result = execute_study_command(StudyCommandRequest(operation="inspect", study_dir=Path("study")))
    encoded = json.dumps(result.as_dict())
    assert result.exit_code == 7
    assert "private secret" not in encoded
    assert "Traceback" not in encoded


def test_cli_parser_contains_no_study_transition_logic() -> None:
    source = Path("src/vamos/experiment/cli/study.py").read_text(encoding="utf-8")
    assert "experiment.study.models" not in source
    assert "experiment.study.loading" not in source
    assert "current_state" not in source
    assert "transition" not in source


def test_study_summarize_human_output_lists_tasks(tmp_path: Path) -> None:
    config = tmp_path / "study.json"
    root = tmp_path / "study"
    _write_spec(config, seeds=[0, 1])
    assert _run("create", str(config), "--output", str(root), "--json").returncode == 0
    assert _run("run", str(root), "--json").returncode == 0

    human = _run("summarize", str(root))

    assert human.returncode == 0, human.stderr
    assert "Tasks:" in human.stdout
    assert sum(" succeeded " in line for line in human.stdout.splitlines()) == 2
    assert "Next: vamos study inspect STUDY_DIR" in human.stdout


def test_every_study_operation_prints_human_output(tmp_path: Path) -> None:
    config = tmp_path / "study.json"
    root = tmp_path / "study"
    _write_spec(config)
    commands = [
        ("plan", ["plan", str(config)]),
        ("create", ["create", str(config), "--output", str(root)]),
        ("inspect", ["inspect", str(root)]),
        ("run", ["run", str(root)]),
        ("resume", ["resume", str(root)]),
        ("summarize", ["summarize", str(root)]),
    ]

    for operation, argv in commands:
        proc = _run(*argv)
        assert proc.returncode == 0, (operation, proc.stdout, proc.stderr)
        assert f"Study {operation}:" in proc.stdout, (operation, proc.stdout)
        assert "Traceback" not in proc.stdout + proc.stderr
    # A completed study has no failed task to retry: a refusal, reported without a traceback.
    retry = _run("retry", str(root), "--failed")
    assert retry.returncode == 4
    assert "Error [study_retry_not_allowed]" in retry.stderr and "Traceback" not in retry.stderr
