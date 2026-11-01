from __future__ import annotations

import json

from vamos.assist.apply import apply_plan
from vamos.assist.plan import create_plan
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


def test_apply_plan_materializes_project(tmp_path) -> None:
    plan_dir = create_plan(
        prompt="Create a runnable project from demo template.",
        template="demo",
        problem_type="real",
        out_dir=tmp_path / "plan",
    )

    project_dir = apply_plan(plan_dir, out_dir=tmp_path / "project")

    assert (project_dir / "config.json").is_file()
    assert (project_dir / "plan.json").is_file()
    assert (project_dir / "prompt.txt").is_file()
    assert (project_dir / "catalog.json").is_file()
    assert (project_dir / "README_run.md").is_file()

    config_data = json.loads((project_dir / "config.json").read_text(encoding="utf-8"))
    validate_experiment_spec(config_data, allowed_overrides=_allowed_overrides())

    readme = (project_dir / "README_run.md").read_text(encoding="utf-8")
    assert "vamos --config config.json" in readme
    assert "python -m vamos.experiment.cli.main --config config.json" in readme


def test_apply_overwrite_refuses_unmarked_or_plan_folder(tmp_path) -> None:
    import pytest

    plan_dir = create_plan(prompt="Overwrite safety.", template="demo", problem_type="real", out_dir=tmp_path / "plan")
    plan_files = {path.name: path.read_bytes() for path in plan_dir.iterdir() if path.is_file()}

    for target in (plan_dir, tmp_path):
        with pytest.raises(ValueError, match="plan directory"):
            apply_plan(plan_dir, out_dir=target, overwrite=True)
    assert {path.name: path.read_bytes() for path in plan_dir.iterdir() if path.is_file()} == plan_files

    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "thesis.tex").write_text("keep me", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.vamos-assist"):
        apply_plan(plan_dir, out_dir=foreign, overwrite=True)
    assert (foreign / "thesis.tex").read_text(encoding="utf-8") == "keep me"

    project = apply_plan(plan_dir, out_dir=tmp_path / "project")
    (project / "notes.md").write_text("mine", encoding="utf-8")
    (project / "config.json").write_text("{}", encoding="utf-8")
    apply_plan(plan_dir, out_dir=project, overwrite=True)
    assert (project / "notes.md").read_text(encoding="utf-8") == "mine"
    assert (project / "config.json").read_bytes() == plan_files["config.json"]
