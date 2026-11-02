from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from vamos.engine.config.spec import EXPERIMENT_SPEC_VERSION
from vamos.experiment.cli import quickstart, quickstart_output


def test_quickstart_template_demo_exists() -> None:
    template = quickstart.get_template("demo")
    assert template.defaults.problem == "zdt1"


def test_quickstart_template_list_smoke(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        quickstart.run_quickstart(["--template", "list", "--yes"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "Quickstart templates:" in out
    assert "demo" in out


def test_quickstart_write_spec(tmp_path) -> None:
    template = quickstart.get_template("demo")
    spec_path = quickstart_output._write_spec(
        title="Quickstart: Demo",
        problem=template.defaults.problem,
        algorithm=template.defaults.algorithm,
        engine=template.defaults.engine,
        budget=template.defaults.budget,
        pop_size=template.defaults.pop_size,
        seed=template.defaults.seed,
        output_root=str(tmp_path / "results"),
        plot=template.defaults.plot,
        config_path=str(tmp_path / "quickstart.json"),
    )
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    assert data["version"] == EXPERIMENT_SPEC_VERSION
    defaults = data["defaults"]
    assert defaults["problem"] == "zdt1"
    assert defaults["algorithm"] == "nsgaii"
    assert defaults["engine"] == "numpy"


@pytest.mark.cli
def test_quickstart_subprocess_smoke(tmp_path) -> None:
    output_root = tmp_path / "results"
    config_path = tmp_path / "quickstart.json"
    env = os.environ.copy()
    env.update({"MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"})
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "vamos.experiment.cli.main",
            "quickstart",
            "--template",
            "demo",
            "--yes",
            "--no-plot",
            "--budget",
            "12",
            "--pop-size",
            "6",
            "--output-root",
            str(output_root),
            "--config-path",
            str(config_path),
            "--no-preflight",
        ],
        capture_output=True,
        timeout=120,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr.decode()
    assert config_path.exists()
    run_dir = output_root / "ZDT1" / "nsgaii" / "numpy" / "seed_42"
    assert {path.name for path in run_dir.iterdir()} == {"manifest.json", "result.npz", "environment.json"}
    assert b"Quickstart complete." in proc.stdout


def _quickstart(output_root, *extra):
    from vamos.experiment.cli import quickstart

    base = ["--template", "demo", "--yes", "--no-plot", "--budget", "12", "--pop-size", "6", "--no-preflight"]
    quickstart.run_quickstart([*base, "--output-root", str(output_root), *extra])


def test_quickstart_twice_does_not_fail(tmp_path, capsys) -> None:
    output_root = tmp_path / "results"
    _quickstart(output_root)
    first = capsys.readouterr().out

    _quickstart(output_root)

    second = capsys.readouterr().out
    assert "seed_42" in first and "seed_43" in second
    assert "Seed 42 already has a run" in second
    assert len(list(output_root.rglob("manifest.json"))) == 2
    assert len(list(output_root.glob("quickstart_*.json"))) == 2
    # The suggested re-run command uses a free seed, so it works.
    rerun = next(line for line in second.splitlines() if line.startswith("Re-run:"))
    assert "--seed 44" in rerun
    assert "disabled by --no-plot" in second


def test_quickstart_with_a_taken_explicit_seed_stops_before_writing(tmp_path, capsys) -> None:
    output_root = tmp_path / "results"
    _quickstart(output_root, "--seed", "5")
    capsys.readouterr()

    with pytest.raises(SystemExit) as excinfo:
        _quickstart(output_root, "--seed", "5")

    out = capsys.readouterr().out
    assert excinfo.value.code == 8
    assert "already exists" in out and "--seed 6" in out
    assert len(list(output_root.glob("quickstart_*.json"))) == 1
