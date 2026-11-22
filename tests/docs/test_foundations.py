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
    assert list(sections) == ["Basic", "Intermediate", "Advanced"]
    # The Intermediate level is grouped by the family of each algorithm, then by the topics that they share.
    groups = [next(iter(item)) for item in sections["Intermediate"]]
    assert groups == ["Dominance-based", "Decomposition-based", "Indicator-based", "Swarm-based", "Operators"]

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


def _run_operator_ports(cases: dict[str, str], n: int) -> dict[str, list[list[float]]]:
    """Children of the figures' operator ports: ``n`` calls of each JavaScript expression."""
    program = f"""
require({json.dumps(str(SCRIPT))});
const V = globalThis.VamosFoundations;
const O = V.operators;
const P1 = [0.35, 0.3], P2 = [0.6, 0.55], P3 = [0.3, 0.7], X = [0.3, 0.6];
const out = {{}};
for (const [name, expr] of Object.entries({json.dumps(cases)})) {{
  const r = V.mulberry32(17);
  const run = new Function("O", "P1", "P2", "P3", "X", "r", "V", "return " + expr);
  const rows = [];
  for (let i = 0; i < {n}; i++) for (const c of run(O, P1, P2, P3, X, r, V)) rows.push(c);
  out[name] = rows;
}}
process.stdout.write(JSON.stringify(out));
"""
    completed = subprocess.run(["node", "-e", program], capture_output=True, text=True, check=True, timeout=120)
    return json.loads(completed.stdout)


@pytest.mark.skipif(shutil.which("node") is None, reason="the figures' script runs under Node.js")
def test_real_operator_ports_sample_the_vamos_distributions() -> None:
    from vamos.engine.operators.impl import real

    n = 20_000
    b = {"lower": np.zeros(2), "upper": np.ones(2)}
    pair = np.tile(np.array([[[0.35, 0.3], [0.6, 0.55]]]), (n, 1, 1))
    group = np.tile(np.array([[[0.35, 0.3], [0.6, 0.55], [0.3, 0.7]]]), (n, 1, 1))
    x = np.tile(np.array([[0.3, 0.6]]), (n, 1))
    cases: dict[str, tuple[str, Any]] = {
        "blx_alpha": ("O.blxAlphaBeta(P1, P2, 0.5, 0.5, r)", lambda g: real.BLXAlphaCrossover(alpha=0.5, prob_crossover=1.0, **b)(pair, g)),
        "blx_alpha_beta": (
            "O.blxAlphaBeta(P1, P2, 0.2, 0.8, r)",
            lambda g: real.BLXAlphaBetaCrossover(alpha=0.2, beta=0.8, prob_crossover=1.0, **b)(pair, g),
        ),
        "arithmetic": ("O.arithmetic(P1, P2, r)", lambda g: real.ArithmeticCrossover(prob_crossover=1.0)(pair, g)),
        "laplace": ("O.laplace(P1, P2, 0.1, 0.5, r)", lambda g: real.LaplaceCrossover(a=0.1, b=0.5, prob_crossover=1.0, **b)(pair, g)),
        "fuzzy": ("O.fuzzy(P1, P2, 0.5, 0, 1, r)", lambda g: real.FuzzyCrossover(d=0.5, prob_crossover=1.0, **b)(pair, g)),
        "de": ("O.deMating(P1, P2, P3, 0.5, 0.6, r)", lambda g: real.DEMatingCrossover(F=0.5, CR=0.6, **b)(group, g)),
        "pcx": ("O.pcx([P1, P2, P3], 0.1, 0.2, r)", lambda g: real.PCXCrossover(sigma_eta=0.1, sigma_zeta=0.2, **b)(group, g)),
        "undx": ("O.undx([P1, P2, P3], 0.5, 0.35, r)", lambda g: real.UNDXCrossover(prob_crossover=1.0, zeta=0.5, eta=0.35, **b)(group, g)),
        "simplex": ("O.spx([P1, P2, P3], null, r)", lambda g: real.SPXCrossover(**b)(group, g)),
        "pm": (
            "[V.polynomialMutation(X.slice(), 0, 1, 1, 20, r)]",
            lambda g: real.PolynomialMutation(prob_mutation=1.0, eta=20.0, **b)(x, g),
        ),
        "gaussian": (
            "[O.gaussianMutation(X, 0, 1, 0.5, 0.1, r)]",
            lambda g: real.GaussianMutation(prob_mutation=0.5, sigma=0.1, **b)(x, g),
        ),
        "uniform": ("[O.uniformMutation(X, 0, 1, 1, 0.2, r)]", lambda g: real.UniformMutation(prob_mutation=1.0, perturb=0.2, **b)(x, g)),
        "uniform_reset": ("[O.uniformReset(X, 0, 1, 1, r)]", lambda g: real.UniformResetMutation(prob_mutation=1.0, **b)(x, g)),
        "non_uniform": ("[O.nonUniform(X, 0, 1, 1, r)]", lambda g: real.NonUniformMutation(prob_mutation=1.0, **b)(x, g)),
        "cauchy": ("[O.cauchyMutation(X, 0, 1, 1, 0.1, r)]", lambda g: real.CauchyMutation(prob_mutation=1.0, gamma=0.1, **b)(x, g)),
        "levy_flight": (
            "[O.levyFlight(X, 0, 1, 1, 1.5, 0.05, r)]",
            lambda g: real.LevyFlightMutation(prob_mutation=1.0, beta=1.5, scale=0.05, **b)(x, g),
        ),
        "power_law": ("[O.powerLaw(X, 0, 1, 1, 3, r)]", lambda g: real.PowerLawMutation(prob_mutation=1.0, index=3.0, **b)(x, g)),
        "linked_polynomial": (
            "[O.linkedPolynomial(X, 0, 1, 1, 5, r)]",
            lambda g: real.LinkedPolynomialMutation(prob_mutation=1.0, eta=5.0, **b)(x, g),
        ),
    }
    ports = _run_operator_ports({name: expr for name, (expr, _) in cases.items()}, n)
    levels = [0.1, 0.25, 0.5, 0.75, 0.9]
    for name, (_, operator) in cases.items():
        expected = np.asarray(operator(np.random.default_rng(20261121)), dtype=float).reshape(-1, 2)
        sampled = np.asarray(ports[name], dtype=float)
        assert sampled.shape == expected.shape, name
        q_js = np.quantile(sampled, levels, axis=0)
        q_py = np.quantile(expected, levels, axis=0)
        # Six standard errors of a sample quantile, sqrt(p (1 - p) / n) / density, with the density
        # estimated from the quantiles 0.02 below and above.
        spread = np.quantile(expected, np.add(levels, 0.02), axis=0) - np.quantile(expected, np.subtract(levels, 0.02), axis=0)
        error = np.sqrt(np.multiply(levels, np.subtract(1, levels)) / len(expected))[:, None] * spread / 0.04
        tolerance = 6 * error + 1e-3
        assert np.all(np.abs(q_js - q_py) <= tolerance), f"{name}: {q_js.tolist()} != {q_py.tolist()}"
    # Linked polynomial mutation: one draw per individual, so both variables move in the same direction.
    steps = np.sign(np.asarray(ports["linked_polynomial"]) - [0.3, 0.6])
    assert np.all(steps[:, 0] == steps[:, 1])
    whole = real.WholeArithmeticCrossover(alpha=0.25, prob_crossover=1.0)(pair[:1], np.random.default_rng(0))
    np.testing.assert_allclose(_run_operator_ports({"whole": "O.wholeArithmetic(P1, P2, 0.25)"}, 1)["whole"], whole[0], rtol=1e-12)


class _Draws:
    """A generator that returns prepared draws, to replay VAMOS operators with known cut points and masks."""

    def __init__(self, *draws: object) -> None:
        self._draws = list(draws)

    def _next(self, *_: object, **__: object) -> Any:
        return self._draws.pop(0)

    random = integers = choice = _next


@pytest.mark.skipif(shutil.which("node") is None, reason="the figures' script runs under Node.js")
def test_discrete_operator_ports_agree_with_vamos() -> None:
    from vamos.engine.operators.impl import _permutation_crossovers as perm_cx
    from vamos.engine.operators.impl import _permutation_mutations as perm_mut
    from vamos.engine.operators.impl import binary

    rng = np.random.default_rng(5)
    exprs: list[str] = []
    expected: list[object] = []

    def check(expr: str, value: object) -> None:
        exprs.append(expr)
        expected.append(np.asarray(value).tolist())

    n = 12
    for _ in range(10):
        p1, p2 = rng.integers(0, 2, n), rng.integers(0, 2, n)
        parents = np.stack([p1, p2]).astype(np.int8)
        a, b = sorted(rng.choice(n, 2, replace=False).tolist())
        keep = rng.random(n) < 0.5
        mask = rng.random(n) < 0.3
        diff = np.flatnonzero(p1 != p2)
        args = f"{json.dumps(p1.tolist())}, {json.dumps(p2.tolist())}"
        check(f"O.onePoint({args}, {b})", binary.one_point_crossover(parents, 1.0, _Draws(np.zeros(1), np.array([b]))))
        check(f"O.twoPoint({args}, {a}, {b})", binary.two_point_crossover(parents, 1.0, _Draws(np.zeros(1), np.array([[a, b]]))))
        uniform = binary.uniform_crossover(parents, 1.0, _Draws(np.zeros(1), np.where(keep, 0.25, 0.75)[None]))
        check(f"O.uniformBits({args}, {json.dumps(keep.tolist())})", uniform)
        if diff.size:
            chosen = np.sort(rng.choice(diff, max(1, diff.size // 2), replace=False))
            check(f"O.hux({args}, {json.dumps(chosen.tolist())})", binary.hux_crossover(parents, 1.0, _Draws(np.zeros(1), chosen)))
        flipped = p1[None].astype(np.int8)
        binary.bit_flip_mutation(flipped, 0.3, _Draws(np.where(mask, 0.1, 0.9)[None]))
        check(f"[O.bitFlip({json.dumps(p1.tolist())}, {json.dumps(mask.tolist())})]", flipped)
        segment = p1[None].astype(np.int8)
        binary.segment_inversion_mutation(segment, 1.0, _Draws(np.zeros(1), a, b - 1))
        check(f"[O.segmentInversion({json.dumps(p1.tolist())}, {a}, {b})]", segment)

    m = 8
    for _ in range(10):
        p1, p2 = rng.permutation(m), rng.permutation(m)
        a, b = sorted(rng.choice(m, 2, replace=False).tolist())
        start = int(rng.integers(m))
        k = int(rng.integers(1, m + 1))
        positions = rng.choice(m, k, replace=False)
        keep = np.isin(np.arange(m), positions)
        s1, s2, sk = (json.dumps(v.tolist()) for v in (p1, p2, keep))
        pmx, ox, inverted, inserted = p1.copy(), p1.copy(), p1.copy(), p1.copy()
        perm_cx._pmx_into(p1, p2, pmx, a, b)
        perm_cx._order_crossover_into(p1, p2, ox, a, b)
        perm_mut._inversion_row_mutation(inverted, _Draws(a, b - 1))
        perm_mut._insert_row_mutation(inserted, _Draws(b, a))
        check(f"[O.pmxChild({s1}, {s2}, {a}, {b})]", [pmx])
        check(f"[O.oxChild({s1}, {s2}, {a}, {b})]", [ox])
        check(f"O.cycle({s1}, {s2}, {start}).children", list(perm_cx._cycle_children(p1, p2, _Draws(start))))
        positional = list(perm_cx._position_based_children(p1, p2, _Draws(k, positions)))
        check(f"[O.positionBased({s1}, {s2}, {sk}), O.positionBased({s2}, {s1}, {sk})]", positional)
        check(f"[O.invert({s1}, {a}, {b})]", [inverted])
        check(f"[O.insert({s1}, {b}, {a})]", [inserted])

    ports = _run_operator_ports({str(i): expr for i, expr in enumerate(exprs)}, 1)
    for i, (expr, value) in enumerate(zip(exprs, expected)):
        assert ports[str(i)] == value, f"{expr}: {ports[str(i)]} != {value}"


@pytest.mark.skipif(shutil.which("node") is None, reason="the figures' script runs under Node.js")
def test_integer_sbx_port_rounds_like_vamos() -> None:
    from vamos.engine.operators.impl.integer import integer_sbx_crossover

    n = 20_000
    parents = np.tile(np.array([[3], [4]]), (n, 1))
    children = integer_sbx_crossover(parents, 1.0, 2.0, np.array([0]), np.array([10]), np.random.default_rng(3))
    ports = np.asarray(_run_operator_ports({"int": "O.integerSbx([3], [4], 0, 10, 2, 0.5, r, false)"}, n)["int"])
    for value in range(11):
        assert abs(np.mean(ports == value) - np.mean(children == value)) <= 0.01, value
    assert abs(ports.mean() - 3.5) <= 0.01


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
