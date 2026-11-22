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


def _run_node(program: str) -> Any:
    """Run a program after the figures' script, through stdin: its data can be too long for a command line."""
    source = f"require({json.dumps(str(SCRIPT))});\n{program}"
    completed = subprocess.run(["node", "-"], input=source, capture_output=True, text=True, check=True, timeout=60)
    return json.loads(completed.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="the figures' script runs under Node.js")
def test_two_layer_lattice_figure_matches_vamos() -> None:
    from vamos.engine.reference_directions import _two_layers, two_layer_shared

    cases = [(2, 4, 2), (3, 4, 2), (3, 6, 1), (3, 6, 2), (3, 12, 3), (5, 3, 2), (8, 3, 2), (10, 3, 2)]
    js = _run_node(f"""
const V = globalThis.VamosFoundations;
process.stdout.write(JSON.stringify({json.dumps(cases)}.map(([m, p1, p2]) => ({{ layers: V.twoLayerLattice(m, p1, p2), shared: V.twoLayerShared(m, p1, p2) }}))));
""")
    for (m, p1, p2), result in zip(cases, js):
        boundary, inside = _two_layers(m, p1, p2)
        np.testing.assert_allclose(result["layers"]["boundary"], boundary, rtol=0, atol=1e-12)
        np.testing.assert_allclose(result["layers"]["inside"], inside, rtol=0, atol=1e-12)
        assert result["shared"] == two_layer_shared(m, p1, p2), (m, p1, p2)


@pytest.mark.skipif(shutil.which("node") is None, reason="the figures' script runs under Node.js")
def test_constraint_handling_figure_ranks_match_vamos() -> None:
    from vamos.engine.algorithm.components.constrained_ranking import (
        pareto_ranks,
        stochastic_ranking_order,
        stochastic_ranking_ranks,
        violation,
    )
    from vamos.engine.algorithm.components.constraint_handling import make_constraint_handling
    from vamos.foundation.kernel.registry import resolve_kernel

    kernel = resolve_kernel("numpy")
    rng = np.random.default_rng(20261121)
    push_pull = {
        "change_threshold": 0.001,
        "window_fraction": 0.02,
        "decay_fraction": 0.8,
        "exponent": 2.0,
        "feasibility_threshold": 0.95,
        "reduction_rate": 0.1,
    }
    STATE = ("pushing", "epsilon")  # what a run teaches a technique
    # The techniques of VAMOS 1.5.2, the published release; stochastic ranking is checked below with its uniforms.
    techniques: list[tuple[str, dict[str, float], dict[str, Any]]] = [
        ("cdp", {}, {}),
        ("violated_constraints", {}, {}),
        ("epsilon_constrained", {"initial_quantile": 0.2, "exponent": 5.0, "decay_fraction": 0.8}, {"epsilon": 0.25}),
        ("push_pull", push_pull, {"pushing": True}),
        ("push_pull", push_pull, {"pushing": False, "epsilon": 0.15}),
        ("self_adaptive_penalty", {}, {}),
    ]
    populations = []
    for size in (6, 10, 14):
        for _ in range(4):
            F = np.round(rng.random((size, 2)), 2)
            G = rng.choice([-0.5, -0.1, 0.0, 0.05, 0.1, 0.2, 0.3], size=(size, 2))
            F[1], G[1] = F[0], G[0]  # a duplicated solution
            populations.append((F, G))
    populations.append((np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]]), np.full((3, 2), -1.0)))  # all feasible
    populations.append((np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]]), np.array([[0.2, 0.0], [0.1, 0.3], [0.2, 0.0]])))  # none feasible
    cases = [
        {
            "name": name,
            "F": F.tolist(),
            "G": G.tolist(),
            "options": {**params, **state},
            "uniforms": rng.random(len(F) * (len(F) - 1)).tolist(),
        }
        for F, G in populations
        for name, params, state in techniques
    ]
    js = _run_node(f"""
const V = globalThis.VamosFoundations;
process.stdout.write(JSON.stringify({json.dumps(cases)}.map((c) => {{
  const sr = V.constraintRanks("stochastic_ranking", c.F, c.G, {{ probability: 0.45, uniforms: c.uniforms }});
  return {{ ranks: V.constraintRanks(c.name, c.F, c.G, c.options).ranks, stochastic: sr.ranks, order: sr.order }};
}})));
""")
    for case, result in zip(cases, js):
        F, G = np.asarray(case["F"]), np.asarray(case["G"])
        params = {key: value for key, value in case["options"].items() if key not in STATE}
        technique: Any = make_constraint_handling((case["name"], params))
        for key in STATE:
            if key in case["options"]:
                setattr(technique, key, case["options"][key])
        expected = technique.ranks(kernel, F, G, np.random.default_rng(0))
        assert result["ranks"] == expected.tolist(), (case["name"], case["options"], F.tolist(), G.tolist())
        pareto = pareto_ranks(kernel, F)
        feasible = np.all(G <= 0.0, axis=1)
        order = stochastic_ranking_order(pareto, feasible, violation(G), np.asarray(case["uniforms"]), 0.45)
        assert result["order"] == order.tolist()
        assert result["stochastic"] == stochastic_ranking_ranks(order, pareto, feasible).tolist()


@pytest.mark.skipif(shutil.which("node") is None, reason="the figures' script runs under Node.js")
def test_evolutionary_loop_and_family_figures_agree_with_vamos() -> None:
    from vamos import select_survivors
    from vamos.engine.algorithm.components.hv_reference import relative_reference_point
    from vamos.engine.algorithm.moead.aggregation import tchebycheff

    rng = np.random.default_rng(20261122)
    cases = [rng.random((int(rng.integers(6, 13)), 2)) for _ in range(12)]
    program = f"""
require({json.dumps(str(SCRIPT))});
const V = globalThis.VamosFoundations;
const loops = ["plus", "comma", "steady"].map((mode) => V.gaLoop(mode, 44, 6, 30));
const families = {json.dumps([case.tolist() for case in cases])}.map((F) => V.familySurvivors(F, 5));
process.stdout.write(JSON.stringify({{ loops, families }}));
"""
    completed = subprocess.run(["node", "-e", program], capture_output=True, text=True, check=True, timeout=60)
    data = json.loads(completed.stdout)

    # The loop figure: N offspring per cycle (one when steady-state), the exact budget, and NSGA-II's survivors.
    for mode, steps in zip(("plus", "comma", "steady"), data["loops"]):
        survive = [step for step in steps if step["phase"] == "survive"]
        assert len(survive) == (24 if mode == "steady" else 4)
        assert steps[-1]["evaluations"] == 30
        for step in survive:
            assert len(step["children"]) == (1 if mode == "steady" else 6)
            ids = [member["id"] for member in step["merged"]]
            kept = {member["id"] for member in step["kept"]}
            if mode == "comma":
                assert kept == {child["id"] for child in step["children"]}
            else:
                F = np.array([member["f"] for member in step["merged"]])
                assert kept == {ids[i] for i in select_survivors(F, survivor_count=6).selected_indices}

    # The family figure: NSGA-II's survivors, MOEA/D's Tchebycheff choices, SMS-EMOA's removals and the leaders.
    for F, js in zip(cases, data["families"]):
        assert js["dominance"] == sorted(int(i) for i in select_survivors(F, survivor_count=5).selected_indices)
        ideal = F.min(axis=0)
        assert js["decomposition"]["best"] == [int(np.argmin(tchebycheff(F, np.array(w), ideal))) for w in js["decomposition"]["W"]]
        alive = list(range(len(F)))
        while len(alive) > 5:
            Q = F[alive]
            last = list(_fast_non_dominated_sort(Q)[0][-1])
            contributions = hypervolume_contributions(Q[last], relative_reference_point(Q, 1.0))
            alive.pop(last[int(np.argmin(contributions))] if len(last) > 1 else last[0])
        assert sorted(js["indicator"]["kept"]) == alive
        assert js["swarm"]["leaders"] == [int(i) for i in np.flatnonzero(_fast_non_dominated_sort(F)[1] == 0)]
