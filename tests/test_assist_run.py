from __future__ import annotations

import json
from pathlib import Path

from pytest import MonkeyPatch

from vamos.assist.apply import apply_plan
from vamos.assist.plan import create_plan
from vamos.assist.run import make_execution_config, run_plan, run_with_config_path, select_config_path
from vamos.engine.config.spec import validate_experiment_spec
from vamos.experiment.cli.args import build_parser, build_pre_parser
from vamos.experiment.cli.loaders import load_spec_defaults
from vamos.experiment.cli.spec_args import parser_spec_keys
from vamos.foundation.core.experiment_config import ExperimentConfig


def _allowed_overrides() -> set[str]:
    default_config = ExperimentConfig()
    parser = build_parser(
        default_config=default_config,
        pre_parser=build_pre_parser(),
        spec_defaults=load_spec_defaults(None),
    )
    return parser_spec_keys(parser)


def _materialized_plan(tmp_path: Path) -> tuple[Path, Path]:
    plan_dir = create_plan(
        prompt="Run plan smoke test.",
        template="demo",
        problem_type="real",
        out_dir=tmp_path / "plan",
    )
    project_dir = apply_plan(plan_dir, out_dir=tmp_path / "project")
    (plan_dir / "project").mkdir(parents=True, exist_ok=True)
    (plan_dir / "project" / "config.json").write_text((project_dir / "config.json").read_text(encoding="utf-8"), encoding="utf-8")
    return plan_dir, project_dir


def test_select_config_path_prefers_project_config(tmp_path: Path) -> None:
    plan_dir, _ = _materialized_plan(tmp_path)
    selected = select_config_path(plan_dir)
    assert selected == plan_dir / "project" / "config.json"


def test_make_execution_config_smoke_is_valid_and_sets_output_root(tmp_path: Path) -> None:
    plan_dir, project_dir = _materialized_plan(tmp_path)
    base_config = json.loads((project_dir / "config.json").read_text(encoding="utf-8"))

    run_dir = plan_dir / "runs" / "run_smoke"
    resolved = make_execution_config(base_config, run_dir=run_dir, smoke=True, smoke_evals=123)

    validate_experiment_spec(resolved, allowed_overrides=_allowed_overrides())
    defaults = resolved["defaults"]
    assert defaults["max_evaluations"] == 123
    assert defaults["output_root"] == str(run_dir / "results")


def test_run_plan_writes_report_with_stubbed_execution(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    plan_dir, _ = _materialized_plan(tmp_path)

    def _fake_run(config_path: Path) -> int:
        assert config_path.is_file()
        return 0

    monkeypatch.setattr("vamos.assist.run.run_with_config_path", _fake_run)

    summary = run_plan(
        plan_dir=plan_dir,
        smoke=True,
        smoke_evals=111,
    )

    assert summary["status"] == "ok"
    run_dir = Path(str(summary["run_dir"]))
    execution_path = run_dir / "execution_config.json"
    report_path = run_dir / "run_report.json"
    assert execution_path.is_file()
    assert report_path.is_file()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    for key in (
        "status",
        "exit_code",
        "plan_dir",
        "base_config_path",
        "execution_config_path",
        "run_dir",
        "started_at",
        "ended_at",
        "smoke",
        "smoke_evals",
        "execution_mode",
        "warnings",
    ):
        assert key in report


def test_run_with_config_path_prefers_in_process(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "execution_config.json"
    config_path.write_text("{}", encoding="utf-8")

    calls = {"in_process": 0, "subprocess": 0}

    def _fake_in_process(path: str) -> int:
        assert path == str(config_path)
        calls["in_process"] += 1
        return 0

    def _fake_subprocess(path: Path) -> int:
        del path
        calls["subprocess"] += 1
        return 1

    monkeypatch.setattr("vamos.experiment.cli.main.run_from_config_path", _fake_in_process)
    monkeypatch.setattr("vamos.assist.run._run_with_subprocess", _fake_subprocess)

    exit_code = run_with_config_path(config_path)
    assert exit_code == 0
    assert calls == {"in_process": 1, "subprocess": 0}


def test_run_with_config_path_falls_back_to_subprocess_only_without_the_cli(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    config_path = tmp_path / "execution_config.json"
    config_path.write_text("{}", encoding="utf-8")
    calls = {"subprocess": 0}

    def _fake_subprocess(path: Path) -> int:
        assert path == config_path
        calls["subprocess"] += 1
        return 0

    monkeypatch.setattr("vamos.assist.run._in_process_runner", lambda: None)
    monkeypatch.setattr("vamos.assist.run._run_with_subprocess", _fake_subprocess)

    assert run_with_config_path(config_path) == 0
    assert calls == {"subprocess": 1}


def test_run_plan_reports_in_process_error_without_rerun(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    plan_dir, _ = _materialized_plan(tmp_path)
    calls = {"in_process": 0}

    def _failing_run(path: str) -> int:
        calls["in_process"] += 1
        raise RuntimeError("objective raised")

    def _no_subprocess(path: Path) -> int:
        raise AssertionError("the run must not be repeated in a subprocess")

    monkeypatch.setattr("vamos.experiment.cli.main.run_from_config_path", _failing_run)
    monkeypatch.setattr("vamos.assist.run._run_with_subprocess", _no_subprocess)

    report = run_plan(plan_dir, smoke=True, smoke_evals=20)

    assert calls == {"in_process": 1}
    assert report["status"] == "error"
    assert "objective raised" in str(report["error_message"])
    assert any("objective raised" in warning for warning in report["warnings"])
    assert json.loads(Path(str(report["run_report_path"])).read_text(encoding="utf-8"))["status"] == "error"


def test_run_plan_writes_report_on_systemexit(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    plan_dir, _ = _materialized_plan(tmp_path)

    def _exiting_run(path: str) -> int:
        raise SystemExit(2)

    monkeypatch.setattr("vamos.experiment.cli.main.run_from_config_path", _exiting_run)

    report = run_plan(plan_dir, smoke=True, smoke_evals=20)

    stored = json.loads(Path(str(report["run_report_path"])).read_text(encoding="utf-8"))
    assert stored["status"] == "error"
    assert stored["exit_code"] == 2


def test_run_plan_overwrite_refuses_unmarked_or_plan_folder(tmp_path: Path) -> None:
    import pytest

    plan_dir, _ = _materialized_plan(tmp_path)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "data.csv").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="plan directory"):
        run_plan(plan_dir, out_dir=plan_dir, overwrite=True, smoke=True)
    with pytest.raises(ValueError, match=r"\.vamos-assist"):
        run_plan(plan_dir, out_dir=foreign, overwrite=True, smoke=True)
    assert (foreign / "data.csv").read_text(encoding="utf-8") == "keep"
    assert (plan_dir / "plan.json").is_file()
