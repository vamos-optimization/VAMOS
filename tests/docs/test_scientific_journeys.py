from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

JOURNEYS = {
    "Try VAMOS": "examples/journeys/try_vamos.py",
    "Solve my problem": "examples/journeys/solve_my_problem.py",
    "Run a reproducible study": "examples/journeys/reproducible_study.py",
}


def test_three_homepage_journeys_have_executable_sources() -> None:
    homepage = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    examples_page = (ROOT / "docs" / "examples.md").read_text(encoding="utf-8")

    assert "examples.md#three-executable-journeys" in homepage
    for label, relative_path in JOURNEYS.items():
        path = ROOT / relative_path
        assert path.is_file(), relative_path
        assert label in examples_page
        assert relative_path in examples_page


def test_journey_scripts_use_public_facades_and_bounded_runs() -> None:
    for relative_path in JOURNEYS.values():
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "from vamos import" in text
        assert "max_evaluations=" in text
        assert "vamos.engine" not in text
        assert "vamos.foundation" not in text
        assert "numpy.random" not in text

    first_run = (ROOT / JOURNEYS["Try VAMOS"]).read_text(encoding="utf-8")
    custom = (ROOT / JOURNEYS["Solve my problem"]).read_text(encoding="utf-8")
    study = (ROOT / JOURNEYS["Run a reproducible study"]).read_text(encoding="utf-8")

    assert 'engine="numpy"' in first_run
    assert "make_problem" in custom
    assert "plan_study" in study
    assert "create_study" in study
    assert "seeds=[0, 1]" in study
