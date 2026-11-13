from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
DOCS = ROOT / "docs"
REPOSITORY_FILE = re.compile(
    r"https://(?:github\.com/vamos-optimization/VAMOS/(?:blob|tree)/main|raw\.githubusercontent\.com/vamos-optimization/VAMOS/main)/(?P<path>[^#?]+)"
)
DOCUMENTATION_PAGE = re.compile(r"https://vamos-optimization\.org/(?P<route>[^#?]*)(?:#(?P<anchor>.+))?")


def _readme_prose() -> str:
    return re.sub(r"```.*?```", "", README.read_text(encoding="utf-8"), flags=re.DOTALL)


def _heading_anchors(page: Path) -> set[str]:
    anchors = set()
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*$", page.read_text(encoding="utf-8"), flags=re.MULTILINE):
        explicit = re.search(r"\{#([^}]+)\}$", heading)
        if explicit:
            anchors.add(explicit.group(1))
            continue
        text = unicodedata.normalize("NFKD", heading).encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"[^\w\s-]", "", text).strip().lower()
        anchors.add(re.sub(r"[-\s]+", "-", text))
    return anchors


def _link_target_exists(link: str) -> bool:
    repository_file = REPOSITORY_FILE.fullmatch(link)
    if repository_file:
        return (ROOT / repository_file.group("path")).exists()
    page = DOCUMENTATION_PAGE.fullmatch(link)
    if not page:
        return True
    route = page.group("route").strip("/")
    candidates = [DOCS / "index.md"] if not route else [DOCS / f"{route}.md", DOCS / route / "index.md"]
    source = next((candidate for candidate in candidates if candidate.is_file()), None)
    if source is None:
        return False
    return page.group("anchor") is None or page.group("anchor") in _heading_anchors(source)


def test_readme_links_are_absolute() -> None:
    """The README is also the PyPI project description, where repository paths do not resolve."""
    prose = _readme_prose()
    links = re.findall(r"!?\[[^\]]*\]\(([^)\s]+)\)", prose)
    assert links

    assert [link for link in links if not link.startswith(("https://", "#"))] == []
    # A path in backticks is not a link on PyPI either; link the page instead.
    assert re.findall(r"`([^`\s]+\.(?:md|cff))`", prose) == []
    assert [link for link in links if not _link_target_exists(link)] == []


def _python_blocks(text: str) -> list[str]:
    return re.findall(r"```python\s*\n(.*?)```", text, flags=re.DOTALL)


@pytest.mark.smoke
def test_readme_study_example_compares_a_quality_indicator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The multi-seed example ranks its seeds by a quality indicator with a fixed reference point.

    Every run spends exactly its budget, so comparing ``evaluations`` across seeds
    compares equal numbers, and the "best run" is the first one.
    """
    (block,) = [source for source in _python_blocks(README.read_text(encoding="utf-8")) if "seed=[" in source]
    assert '"evaluations"' not in block, block

    tree = ast.parse(block)
    reference = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "reference_point" for target in node.targets)
    ]
    assert len(reference) == 1, block
    # A literal point, not one derived from the fronts it measures.
    assert {node.id for node in ast.walk(reference[0]) if isinstance(node, ast.Name)} <= {"np"}

    pytest.importorskip("moocore")
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, Any] = {"__name__": "__main__"}
    exec(compile(block, "README.md::study-example", "exec"), namespace)

    study = namespace["study"]
    hv = np.asarray(namespace["hv"], dtype=float)
    assert hv.shape == (len(study),)
    assert np.all(hv > 0.0)
    assert len(np.unique(hv)) == len(study)


def _run_vamos(*args: str, timeout: int = 180) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update({"MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"})
    return subprocess.run(
        [sys.executable, "-m", "vamos.experiment.cli.main", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


@pytest.mark.smoke
def test_readme_quickstart_template_list_command_smoke() -> None:
    source_path = "README.md"
    proc = _run_vamos("quickstart", "--template", "list", "--yes")
    assert proc.returncode == 0, f"{source_path}: {proc.stderr or proc.stdout}"
    output = (proc.stderr + proc.stdout).lower()
    assert "demo" in output, source_path


@pytest.mark.smoke
def test_readme_create_problem_command_smoke(tmp_path: Path) -> None:
    source_path = "README.md"
    output_file = tmp_path / "readme_problem.py"
    proc = _run_vamos("create-problem", "--yes", "--output", str(output_file))
    assert proc.returncode == 0, f"{source_path}: {proc.stderr or proc.stdout}"
    assert output_file.exists(), source_path


@pytest.mark.smoke
def test_readme_profile_command_smoke(tmp_path: Path) -> None:
    source_path = "README.md"
    output_csv = tmp_path / "profile.csv"
    proc = _run_vamos(
        "profile",
        "--problem",
        "zdt1",
        "--engines",
        "numpy",
        "--budget",
        "200",
        "--repeats",
        "1",
        "--no-hv",
        "--output",
        str(output_csv),
    )
    assert proc.returncode == 0, f"{source_path}: {proc.stderr or proc.stdout}"
    assert output_csv.exists(), source_path


@pytest.mark.smoke
def test_readme_bench_smoke_command_smoke(tmp_path: Path) -> None:
    source_path = "README.md"
    output_dir = tmp_path / "report"
    proc = _run_vamos(
        "bench",
        "ZDT_small",
        "--algorithms",
        "nsgaii",
        "--output",
        str(output_dir),
        "--smoke",
        timeout=240,
    )
    assert proc.returncode == 0, f"{source_path}: {proc.stderr or proc.stdout}"
    assert (output_dir / "summary" / "metrics.csv").exists(), source_path


@pytest.mark.smoke
def test_readme_tune_smoke_command_smoke(tmp_path: Path) -> None:
    source_path = "README.md"
    output_dir = tmp_path / "tuning_smoke"
    proc = _run_vamos(
        "tune",
        "--instances",
        "zdt1,zdt2,zdt3,dtlz1,dtlz2,wfg1",
        "--algorithm",
        "nsgaii",
        "--backend",
        "random",
        "--smoke",
        "--output-dir",
        str(output_dir),
        "--name",
        "readme_tune_smoke",
        timeout=180,
    )
    assert proc.returncode == 0, f"{source_path}: {proc.stderr or proc.stdout}"
    assert (output_dir / "readme_tune_smoke" / "tuning_summary.json").exists(), source_path


def test_readme_api_contracts_describe_the_hypervolume_and_population_rules() -> None:
    from vamos.algorithms import NSGAIIIConfig
    from vamos.foundation.quality_indicators.hypervolume import hypervolume

    readme = README.read_text(encoding="utf-8")
    assert "requires the reference point to dominate all points" not in readme
    assert "require `pop_size` to match" not in readme

    # What the README states: a point the reference does not dominate is left out, and the flag only warns.
    with pytest.warns(FutureWarning):
        assert hypervolume(np.array([[0.5, 0.5], [2.0, 0.1]]), np.array([1.0, 1.0]), allow_ref_expand=True) == pytest.approx(0.25)
    # NSGA-III takes more individuals than its 91 directions for three objectives.
    assert NSGAIIIConfig.default(n_obj=3, pop_size=100).pop_size == 100
