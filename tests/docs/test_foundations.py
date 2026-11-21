"""The Foundations learning path: navigation, metadata, figures, examples and source excerpts."""

from __future__ import annotations

import contextlib
import io
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from vamos.foundation.kernel.numpy_backend import _compute_crowding, _fast_non_dominated_sort
from vamos.foundation.quality_indicators.hypervolume import hypervolume, hypervolume_contributions

from .smoke_manifest import DOC_SMOKE_CASES

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
FOUNDATIONS = DOCS / "foundations"
SCRIPT = DOCS / "javascripts" / "foundations.js"
PAGES = sorted(FOUNDATIONS.rglob("*.md"))
LESSONS = [page for page in PAGES if page.name != "index.md"]
FENCE = re.compile(r"^```(?P<lang>[\w-]*)\n(?P<body>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)
EXCERPT = re.compile(r"\A# Excerpt from (?P<path>\S+)\n")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?")


def _relative(page: Path) -> str:
    return page.relative_to(DOCS).as_posix()


def _nav_pages(node: object) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        return [page for item in node for page in _nav_pages(item)]
    if isinstance(node, dict):
        return [page for value in node.values() for page in _nav_pages(value)]
    return []


def _config() -> dict[str, object]:
    return yaml.safe_load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"))


def _foundations_nav() -> list[object]:
    nav = _config()["nav"]
    assert isinstance(nav, list)
    return next(item["Foundations"] for item in nav if isinstance(item, dict) and "Foundations" in item)


def _blocks(text: str) -> list[tuple[str, str, int, int]]:
    return [(m["lang"], m["body"], m.start(), m.end()) for m in FENCE.finditer(text)]


def _examples(page: Path) -> list[tuple[str, str]]:
    """Each runnable Python block with the text block that documents its output."""
    text = page.read_text(encoding="utf-8")
    blocks = _blocks(text)
    examples = []
    for index, (lang, body, _, end) in enumerate(blocks):
        if lang != "python" or EXCERPT.match(body):
            continue
        assert index + 1 < len(blocks), f"{_relative(page)}: an example has no documented output"
        next_lang, next_body, next_start, _ = blocks[index + 1]
        between = text[end:next_start].strip()
        assert next_lang == "text" and between == "Output:", f"{_relative(page)}: an example must be followed by 'Output:' and a text block"
        examples.append((body, next_body))
    return examples


def _excerpts() -> list[tuple[str, str, str]]:
    found = []
    for page in PAGES:
        for lang, body, _, _ in _blocks(page.read_text(encoding="utf-8")):
            match = EXCERPT.match(body)
            if lang == "python" and match:
                found.append((_relative(page), match["path"], body[match.end() :]))
    return found


def test_foundations_follows_developer_with_levels_and_topics() -> None:
    nav = _config()["nav"]
    titles = [next(iter(item)) for item in nav if isinstance(item, dict)]
    assert titles[-2:] == ["Developer", "Foundations"]

    foundations = _foundations_nav()
    assert foundations[0] == {"Overview": "foundations/index.md"}
    sections = {next(iter(item)): next(iter(item.values())) for item in foundations[1:]}
    assert list(sections) == ["Basic", "Intermediate", "Operators", "Advanced"]
    # The Intermediate level is grouped by the family of each algorithm.
    families = [next(iter(item)) for item in sections["Intermediate"]]
    assert families == ["Dominance-based", "Decomposition-based", "Indicator-based", "Swarm-based"]

    listed = _nav_pages(foundations)
    assert sorted(listed) == sorted(_relative(page) for page in PAGES)
    all_pages = _nav_pages(nav)
    assert [page for page in listed if all_pages.count(page) != 1] == []


def test_assets_are_loaded_by_the_site() -> None:
    config = _config()
    assert "stylesheets/foundations.css" in config["extra_css"]
    assert "javascripts/foundations.js" in config["extra_javascript"]


@pytest.mark.parametrize("page", PAGES, ids=_relative)
def test_pages_have_metadata_one_title_and_sources(page: Path) -> None:
    text = page.read_text(encoding="utf-8")
    metadata = yaml.safe_load(text.split("---", 2)[1])
    assert isinstance(metadata.get("description"), str) and metadata["description"].strip()

    prose = FENCE.sub("", text.split("---", 2)[2])
    assert len(re.findall(r"^# ", prose, flags=re.MULTILINE)) == 1
    if page in LESSONS:
        assert "## Key points" in prose
        assert "## References" in prose
        groups = [[int(n) for n in group.split(", ")] for group in re.findall(r"\[(\d+(?:, \d+)*)\]", prose)]
        cited = {n for group in groups if 0 not in group for n in group}  # [0, 1] is an interval
        listed = {int(n) for n in re.findall(r"^\[(\d+)\] ", prose, flags=re.MULTILINE)}
        assert cited <= listed, f"cited but not listed: {sorted(cited - listed)}"


def test_every_figure_is_implemented_and_has_a_fallback() -> None:
    script = SCRIPT.read_text(encoding="utf-8")
    implemented = set(re.findall(r'widgets\["([\w-]+)"\] = ', script)) | set(re.findall(r"widgets\.(\w+) = ", script))
    used: set[str] = set()
    for page in PAGES:
        for name, inner in re.findall(
            r'<div class="vf-widget" data-vf-widget="([\w-]+)"[^>]*>(.*?)</div>', page.read_text(encoding="utf-8")
        ):
            assert 'class="vf-fallback"' in inner, f"{_relative(page)}: {name} needs a fallback paragraph"
            used.add(name)
    assert used and used <= implemented, sorted(used - implemented)
    assert implemented <= used, f"figures implemented but never used: {sorted(implemented - used)}"


@pytest.mark.parametrize("excerpt", _excerpts(), ids=lambda item: f"{item[0]}:{item[1]}")
def test_source_excerpts_are_verbatim_vamos_code(excerpt: tuple[str, str, str]) -> None:
    page, source, body = excerpt
    lines = (ROOT / source).read_text(encoding="utf-8").splitlines()
    position = 0
    for line in (line for line in body.splitlines() if line.strip()):
        try:
            position = lines.index(line, position) + 1
        except ValueError:
            pytest.fail(f"{page}: line not found in order in {source}: {line!r}")


@pytest.mark.parametrize("page", [page for page in PAGES if _examples(page)], ids=_relative)
def test_examples_run_and_print_the_documented_output(page: Path) -> None:
    for code, documented in _examples(page):
        if "moocore" in code:
            pytest.importorskip("moocore")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(code, str(page), "exec"), {"__name__": "__foundations_example__"})  # noqa: S102
        printed = stdout.getvalue()
        if "optimize(" in code:
            # An optimization run is repeatable in one environment, not bitwise identical
            # everywhere: compare the printed structure and leave the digits free.
            assert NUMBER.sub("#", printed) == NUMBER.sub("#", documented), printed
        else:
            assert printed == documented, printed


def _run_figure_algorithms(cases: list[np.ndarray], ref: list[float]) -> list[dict[str, Any]]:
    """Evaluate the figures' JavaScript algorithms on each set with Node.js."""
    program = f"""
require({json.dumps(str(SCRIPT))});
const V = globalThis.VamosFoundations;
const ref = {json.dumps(ref)};
const results = {json.dumps([case.tolist() for case in cases])}.map((F) => {{
  const sorted = V.nonDominatedSort(F);
  return {{
    rank: sorted.rank,
    crowding: sorted.fronts.map((front) => V.crowdingDistance(F, front).map((d) => (Number.isFinite(d) ? d : null))),
    fronts: sorted.fronts,
    hv: V.hypervolume2d(F, ref),
    contributions: V.hvContributions2d(F, ref).contributions,
  }};
}});
process.stdout.write(JSON.stringify(results));
"""
    completed = subprocess.run(["node", "-e", program], capture_output=True, text=True, check=True, timeout=60)
    return json.loads(completed.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="the figures' script runs under Node.js")
def test_figure_algorithms_agree_with_vamos() -> None:
    rng = np.random.default_rng(20261119)
    ref = [1.1, 1.1]
    cases = [rng.random((int(rng.integers(4, 15)), 2)) for _ in range(20)]
    for case in cases[::4]:
        case[1] = case[0]  # duplicated points
    cases.append(np.array([[0.1, 0.1], [0.5, 0.5]]))  # a dominated point
    cases.append(np.array([[0.2, 0.6], [0.2, 0.6], [0.6, 0.2]]))  # a duplicate next to another point
    cases.append(np.array([[0.2, 0.6], [0.6, 0.2], [1.2, 0.1], [0.3, 1.3]]))  # points outside the reference box
    for F, js in zip(cases, _run_figure_algorithms(cases, ref)):
        fronts, ranks = _fast_non_dominated_sort(F)
        assert js["rank"] == ranks.tolist()
        crowding = _compute_crowding(F, fronts)
        for front, values in zip(js["fronts"], js["crowding"]):
            expected = crowding[np.asarray(front)]
            np.testing.assert_allclose([np.inf if v is None else v for v in values], expected, rtol=1e-12)
        assert js["hv"] == pytest.approx(hypervolume(F, ref), rel=1e-12)
        np.testing.assert_allclose(js["contributions"], hypervolume_contributions(F, ref), rtol=1e-12, atol=1e-15)


@pytest.mark.skipif(shutil.which("node") is None, reason="the figures' script runs under Node.js")
def test_tchebycheff_contour_is_a_level_set() -> None:
    weights = [0.2, 0.35, 0.5, 0.65, 0.8]
    program = f"""
require({json.dumps(str(SCRIPT))});
const V = globalThis.VamosFoundations;
process.stdout.write(JSON.stringify({json.dumps(weights)}.map((w) => V.tchebycheffContour(w, 0.3))));
"""
    completed = subprocess.run(["node", "-e", program], capture_output=True, text=True, check=True, timeout=60)
    for w, contour in zip(weights, json.loads(completed.stdout)):
        vertices = np.asarray(contour, dtype=float)
        # Every vertex and every segment midpoint lies on max(w·f₁, (1 − w)·f₂) = c.
        points = np.vstack([vertices, (vertices[:-1] + vertices[1:]) / 2])
        values = np.maximum(w * points[:, 0], (1 - w) * points[:, 1])
        np.testing.assert_allclose(values, 0.3, rtol=1e-12, err_msg=f"w = {w}: {contour}")


def test_every_page_with_examples_is_a_docs_smoke_case() -> None:
    smoked = {case.source_path for case in DOC_SMOKE_CASES}
    pages = {f"docs/{_relative(page)}" for page in PAGES if _examples(page)}
    assert pages <= smoked, f"Foundations pages with examples missing from tests/docs/smoke_manifest.py: {sorted(pages - smoked)}"


def _run_figure_program(body: str, data: object) -> Any:
    """Evaluate ``body`` with the figures' algorithms as ``V`` and ``data`` read from standard input."""
    program = f"""
require({json.dumps(str(SCRIPT))});
const V = globalThis.VamosFoundations;
const data = JSON.parse(require("fs").readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify({body}));
"""
    completed = subprocess.run(["node", "-e", program], input=json.dumps(data), capture_output=True, text=True, check=True, timeout=60)
    return json.loads(completed.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="the figures' script runs under Node.js")
def test_figure_frace_agrees_with_the_racing_tuner() -> None:
    import logging

    from vamos.engine.tuning import Categorical, Instance, ParamSpace, RacingTuner, Scenario, TuningTask
    from vamos.engine.tuning.racing.stats import select_configs_by_frace

    rng = np.random.default_rng(20261121)
    tests = []
    for _ in range(120):
        scores = rng.normal(0, 1, (int(rng.integers(2, 9)), 1)) + rng.normal(0, 1, (1, int(rng.integers(2, 13))))
        scores = scores + rng.normal(0, 1, scores.shape)
        if rng.random() < 0.3:
            scores = np.round(scores)  # ties within blocks and between differences
        tests.append((scores, bool(rng.random() < 0.5), float(rng.choice([0.01, 0.05, 0.1]))))
    keeps = _run_figure_program("data.map(([S, mx, a]) => V.fraceTest(S, mx, a).keep)", [[s.tolist(), mx, a] for s, mx, a in tests])
    for (scores, maximize, alpha), keep in zip(tests, keeps):
        assert keep == select_configs_by_frace(scores, maximize, alpha).tolist()

    races = []
    for _ in range(20):
        n, n_instances, n_seeds = int(rng.integers(3, 9)), int(rng.integers(2, 5)), int(rng.integers(2, 4))
        scores = rng.normal(0, 1, (n, 1)) + rng.normal(0, 1.2, (n, n_instances * n_seeds))
        races.append((scores, n_instances, n_seeds, float(rng.choice([0.05, 0.1])), int(rng.integers(2, 7))))
    results = _run_figure_program(
        "data.map(([S, a, ft]) => V.fraceRace(S, { alpha: a, firstTest: ft }))", [[s.tolist(), a, ft] for s, _, _, a, ft in races]
    )
    logging.disable(logging.WARNING)
    try:
        for (scores, n_instances, n_seeds, alpha, first_test), js in zip(races, results):
            n = scores.shape[0]
            task = TuningTask(
                name="figure",
                param_space=ParamSpace(params={"c": Categorical("c", list(range(n)))}),
                instances=[Instance(f"i{k}", n_var=1) for k in range(n_instances)],
                seeds=list(range(n_seeds)),
                budget_per_run=1,
            )
            # Unshuffled, the race plays every instance with the first seed, then with the next one.
            scenario = Scenario(
                max_experiments=10**6,
                alpha=alpha,
                first_test=first_test,
                instance_order_random=False,
                seed_order_random=False,
                verbose=False,
            )
            tuner = RacingTuner(task, scenario, seed=0, max_initial_configs=n, initial_configs=[{"c": c} for c in range(n)])

            def score(cfg: dict[str, Any], ctx: Any, scores: np.ndarray = scores, n_instances: int = n_instances) -> float:
                return float(scores[cfg["c"], ctx.seed * n_instances + int(ctx.instance.name[1:])])

            best, history = tuner.run(score, verbose=False)
            assert js["played"] == [trial.details["num_evals"] for trial in history]
            assert js["alive"] == [trial.details["alive"] for trial in history]
            assert js["best"] == best["c"]
    finally:
        logging.disable(logging.NOTSET)


@pytest.mark.skipif(shutil.which("node") is None, reason="the figures' script runs under Node.js")
def test_figure_budgets_agree_with_the_model_based_backends() -> None:
    from vamos.engine.tuning import Instance, ModelBasedTuner, ParamSpace, Real, TuningTask
    from vamos.engine.tuning._model_backend_utils import hyperband_brackets

    cases = [(rungs, eta, configs) for rungs in range(1, 6) for eta in (2, 3, 4) for configs in (1, 5, 17, 60)]
    assert _run_figure_program("data.map(([m, e, c]) => V.hyperbandBrackets(m, e, c))", cases) == [
        hyperband_brackets(*case) for case in cases
    ]

    data, expected = [], []
    for levels in ([1000, 3000, 9000], [500, 5000], [1000, 2000, 4000, 8000]):
        for n_instances, n_seeds in ((1, 1), (6, 5), (7, 4)):
            for fraction in (1.0, 0.75, 0.5, 0.2):
                for min_seeds in (None, 1, 2):
                    task = TuningTask(
                        name="figure",
                        param_space=ParamSpace(params={"x": Real("x", 0.0, 1.0)}),
                        instances=[Instance(f"zdt{k}", n_var=1) for k in range(n_instances)],
                        seeds=list(range(n_seeds)),
                        budget_per_run=levels[-1],
                    )
                    tuner = ModelBasedTuner(
                        task=task,
                        max_trials=1,
                        budget_levels=levels,
                        fidelity_min_instance_frac=fraction,
                        fidelity_min_seed_count=min_seeds,
                    )
                    for budget in sorted({*levels, *(levels[-1] // 3**k for k in range(4)), int(levels[0] * 1.7)}):
                        meta = tuner._resolve_fidelity_slice(budget)[2]
                        expected.append([meta["instances_used"], meta["seeds_used"]])
                        options = {"instances": n_instances, "seeds": n_seeds, "minInstanceFrac": fraction, "minSeedCount": min_seeds or 0}
                        data.append([budget, levels, options])
    assert (
        _run_figure_program("data.map(([b, lv, o]) => { const r = V.fidelitySlice(b, lv, o); return [r.instances, r.seeds]; })", data)
        == expected
    )
