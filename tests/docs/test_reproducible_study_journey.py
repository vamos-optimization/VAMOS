from __future__ import annotations

import itertools
import re
import runpy
from pathlib import Path

import pytest

from vamos import load_study

ROOT = Path(__file__).resolve().parents[2]
HOME = ROOT / "docs" / "index.md"
GUIDE = ROOT / "docs" / "guide" / "studies.md"
DEV_GUIDE = ROOT / "docs" / "dev" / "studies.md"
QUICKSTART = ROOT / "docs" / "guide" / "zero_to_hero.md"
EXAMPLES = ROOT / "docs" / "examples.md"
EXAMPLE = ROOT / "examples" / "journeys" / "reproducible_study.py"


def test_executable_journey_plans_budget_and_traces_every_run(tmp_path: Path) -> None:
    example = runpy.run_path(str(EXAMPLE))
    output = tmp_path / "study"

    preview, completed, report, summary = example["run"](output)

    seeds = tuple(example["SEEDS"])
    max_evaluations = int(example["MAX_EVALUATIONS"])
    expected_combinations = set(itertools.product(preview.problem_ids, preview.algorithm_ids, preview.seeds))

    assert preview.status == "ready"
    assert preview.seeds == seeds
    assert preview.task_count == len(expected_combinations) == 8
    assert preview.total_evaluation_budget == len(expected_combinations) * max_evaluations == 640
    assert completed.plan_id == preview.plan_id

    assert report.state == "completed"
    assert report.counts["tasks"] == 8
    assert report.counts["succeeded"] == 8
    assert report.counts["failed"] == 0
    assert report.verified_run_count == 8
    assert not report.issues

    assert len(summary.rows) == 8
    observed_combinations = {(row.problem_id, row.algorithm_id, row.seed) for row in summary.rows}
    assert observed_combinations == expected_combinations

    for row in summary.rows:
        assert row.state == "succeeded"
        assert row.evaluation_budget == max_evaluations
        assert row.evaluations is not None
        assert 0 < row.evaluations <= max_evaluations
        assert row.selected_run_id is not None
        assert row.run_manifest_path is not None
        assert row.run_metadata_available
        assert row.run_manifest_sha256 is not None
        assert (output / row.run_manifest_path).is_file()


def _python_blocks(text: str) -> list[str]:
    return re.findall(r"```python\s*\n(.*?)```", text, flags=re.DOTALL)


def _section(text: str, heading: str) -> str:
    start = text.index(f"\n{heading}\n")
    end = text.find("\n## ", start + len(heading) + 2)
    return text[start : end if end != -1 else len(text)]


@pytest.mark.smoke
def test_retry_recipe_runs_without_failed_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every task of the documented study succeeds, so the recipe meets the usual case with nothing to retry.

    ``retry`` refuses a study without a retryable failed task (``TERMINAL_TASK_NOT_RETRYABLE``,
    exit code 4 in the CLI); the recipe must check before calling it.
    """
    guide = GUIDE.read_text(encoding="utf-8")
    (tmp_path / "guide").mkdir()
    (tmp_path / "dev").mkdir()
    monkeypatch.chdir(tmp_path / "guide")
    namespace = {"__name__": "__main__"}
    for heading in ("## Python lifecycle in one block", "## Resume and retry"):
        for index, source in enumerate(_python_blocks(_section(guide, heading)), start=1):
            exec(compile(source, f"docs/guide/studies.md::{heading[3:]}::{index}", "exec"), namespace)
    report = load_study("studies/example").inspect()
    assert report.counts["succeeded"] == report.counts["tasks"] == 8
    assert not report.retryable_failed_work

    (block,) = [source for source in _python_blocks(DEV_GUIDE.read_text(encoding="utf-8")) if ".retry(" in source]
    # The contributor example uses a realistic budget; one generation is enough here.
    assert "max_evaluations=10_000" in block
    monkeypatch.chdir(tmp_path / "dev")
    exec(compile(block.replace("max_evaluations=10_000", "max_evaluations=200"), "docs/dev/studies.md", "exec"), {"__name__": "__main__"})
    assert load_study("studies/example").inspect().counts["succeeded"] == 4


def test_learning_path_distinguishes_reproducibility_from_inference() -> None:
    home = HOME.read_text(encoding="utf-8")
    guide = GUIDE.read_text(encoding="utf-8")
    quickstart = QUICKSTART.read_text(encoding="utf-8")
    examples = EXAMPLES.read_text(encoding="utf-8")

    assert "2 × 2 × 2 = 8" in guide
    assert "maximum planned budget" in guide
    assert "same seed list" in guide
    assert "does **not** by itself create common random numbers" in guide
    assert "not automatically a scientifically adequate study" in guide
    assert "selected_run_id" in guide
    assert "run_manifest_path" in guide
    assert "run_manifest_sha256" in guide
    assert "statistical conclusions" in guide
    assert "does not replace experimental-design judgment" in guide

    assert "preview.total_evaluation_budget" in quickstart
    assert "assert study.plan_id == preview.plan_id" in quickstart
    assert "not a publication-grade experimental design" in quickstart

    assert "2 problems × 2 algorithms × 2 seeds" in examples
    assert "not for making comparative performance claims" in examples
    assert "Plan a fair study, measure saved results, and build a report." in home
    assert 'href="tutorials/t06-fair-study/"' in home
    lesson = (ROOT / "docs/tutorials/t06-fair-study.md").read_text(encoding="utf-8")
    assert "../guide/studies.md" in lesson
