from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

import vamos

ROOT = Path(__file__).resolve().parents[2]
PRIMARY_DOCUMENTS = (
    "README.md",
    "docs/guide/installation.md",
    "docs/guide/getting-started.md",
    "docs/guide/run-artifacts.md",
    "docs/guide/studies.md",
    "docs/guide/zero_to_hero.md",
    "docs/guide/studio.md",
    "docs/project/stability-and-versioning.md",
)


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update({"MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"})
    return subprocess.run(
        [sys.executable, "-m", "vamos.experiment.cli.main", *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _python_blocks(document: str) -> list[str]:
    return re.findall(r"```python\s*\n(.*?)```", document, flags=re.DOTALL)


def test_primary_documents_have_no_obsolete_release_or_internal_examples() -> None:
    combined = "\n".join(_read(path) for path in PRIMARY_DOCUMENTS)

    assert "1.5.0" not in combined
    assert "1.1.0" not in combined
    assert "algorithm_kwargs" not in combined
    discarded_study_symbols = ("Study" + "Runner", "Study" + "Task", "run" + "_study")
    assert not any(symbol in combined for symbol in discarded_study_symbols)
    assert "vamos bench --suite" not in combined
    assert re.search(r"from vamos\.(?:engine|experiment|foundation|ux)(?:\.| import)", combined) is None


def test_every_primary_python_snippet_is_valid_python() -> None:
    for path in PRIMARY_DOCUMENTS:
        for index, source in enumerate(_python_blocks(_read(path)), start=1):
            ast.parse(source, filename=f"{path}::python-block-{index}")


@pytest.mark.smoke
def test_primary_optimization_examples_execute() -> None:
    direct = vamos.optimize(
        "zdt1",
        algorithm="nsgaii",
        max_evaluations=16,
        pop_size=8,
        engine="numpy",
        seed=42,
    )
    multi_seed = vamos.optimize("zdt1", algorithm="nsgaii", max_evaluations=16, pop_size=8, seed=[0, 1])
    custom = vamos.make_problem(
        lambda x: [x[0], (1 + x[1]) * (1 - x[0] ** 0.5)],
        n_var=2,
        n_obj=2,
        bounds=[(0, 1), (0, 1)],
        encoding="real",
    )
    custom_result = vamos.optimize(custom, algorithm="nsgaii", max_evaluations=16, pop_size=8, seed=42)

    from vamos.algorithms import NSGAIIConfig
    from vamos.problems import ZDT1

    problem = ZDT1(n_var=6)
    configuration = NSGAIIConfig.default(pop_size=8, n_var=problem.n_var)
    configured = vamos.optimize(
        problem,
        algorithm="nsgaii",
        algorithm_config=configuration,
        max_evaluations=16,
        seed=42,
    )

    assert direct.data["evaluations"] == 16
    assert len(multi_seed.runs) == 2
    assert custom_result.F.shape[1] == 2
    assert configured.data["evaluations"] == 16


@pytest.mark.smoke
def test_primary_run_lifecycle_example(tmp_path: Path) -> None:
    source = tmp_path / "source"
    replay_root = tmp_path / "replay"
    result = vamos.optimize(
        "zdt1",
        algorithm="nsgaii",
        pop_size=8,
        max_evaluations=16,
        engine="numpy",
        seed=7,
        n_var=6,
    )

    stored = vamos.save_result(result, source)
    loaded = vamos.load_result(stored.root)
    run = vamos.load_run(stored.root)
    verification = vamos.verify_run(stored.root, require_level="exact")
    replay = vamos.reproduce(stored.root, output=replay_root)

    assert loaded.F is not None
    assert run.status == "succeeded"
    assert verification.effective_replayability == "exact"
    assert replay.exact is True


@pytest.mark.smoke
def test_primary_study_lifecycle_example(tmp_path: Path) -> None:
    root = tmp_path / "study"
    spec = vamos.StudySpec(
        problems=["zdt1"],
        algorithms=["nsgaii"],
        seeds=[0],
        max_evaluations=16,
        pop_size=8,
        on_error="continue",
    )

    plan = vamos.plan_study(spec, output=root)
    assert not root.exists()
    completed = vamos.create_study(spec, output=root).run()
    loaded = vamos.load_study(root)

    assert plan.plan_id == completed.plan_id
    assert loaded.inspect().counts["succeeded"] == 1
    assert len(loaded.summarize().rows) == 1


@pytest.mark.smoke
@pytest.mark.parametrize(
    "arguments",
    (
        ("results", "inspect", "--help"),
        ("results", "verify", "--help"),
        ("reproduce", "--help"),
        ("study", "plan", "--help"),
        ("study", "create", "--help"),
        ("study", "run", "--help"),
        ("study", "inspect", "--help"),
        ("study", "resume", "--help"),
        ("study", "retry", "--help"),
        ("study", "summarize", "--help"),
    ),
)
def test_documented_stable_commands_exist(arguments: tuple[str, ...]) -> None:
    completed = _run_cli(*arguments)

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "usage:" in completed.stdout.lower()
