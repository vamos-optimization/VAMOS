"""The Foundations learning path: navigation, metadata, figures and examples of how to use VAMOS."""

from __future__ import annotations

import ast
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
EXCERPT = "# Excerpt from "
INTERNAL_PACKAGES = ("vamos.engine", "vamos.foundation", "vamos.experiment")
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
        if lang != "python" or body.startswith(EXCERPT):
            continue  # reported by test_code_shows_how_to_use_vamos_not_how_it_is_implemented
        assert index + 1 < len(blocks), f"{_relative(page)}: an example has no documented output"
        next_lang, next_body, next_start, _ = blocks[index + 1]
        between = text[end:next_start].strip()
        assert next_lang == "text" and between == "Output:", f"{_relative(page)}: an example must be followed by 'Output:' and a text block"
        examples.append((body, next_body))
    return examples


def _vamos_imports(code: str) -> list[tuple[str, list[str]]]:
    """Each VAMOS module that ``code`` imports, with the names it takes from it."""
    found = []
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "vamos":
            found.append((node.module, [alias.name for alias in node.names]))
        elif isinstance(node, ast.Import):
            found.extend((alias.name, []) for alias in node.names if alias.name.split(".")[0] == "vamos")
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
    assert groups == ["Dominance-based", "Decomposition-based", "Indicator-based", "Swarm-based", "Operators", "Benchmarks"]

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


@pytest.mark.parametrize("page", PAGES, ids=_relative)
def test_code_shows_how_to_use_vamos_not_how_it_is_implemented(page: Path) -> None:
    for lang, body, _, _ in _blocks(page.read_text(encoding="utf-8")):
        if lang != "python":
            continue
        assert not body.startswith(EXCERPT), f"{_relative(page)}: show how to use VAMOS, not excerpts of its source"
        for module, names in _vamos_imports(body):
            internal = module.startswith(tuple(f"{package}." for package in INTERNAL_PACKAGES)) or module in INTERNAL_PACKAGES
            private = [part for part in [*module.split("."), *names] if part.startswith("_")]
            assert not internal and not private, f"{_relative(page)}: import the public interface, not {module} {names}"


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

    # Hyperband's budgets: hpbandster's geometric spacing (SMAC3 spaces them alike), rounded to whole evaluations
    # and clipped to the levels as run_smac3 and run_bohb_native round each budget before a run.
    def backend_budgets(low: int, high: int, eta: int) -> list[int]:
        rungs = -int(np.log(low / high) / np.log(eta)) + 1
        spaced = high * np.power(float(eta), -np.linspace(rungs - 1, 0, rungs))
        return [min(high, max(low, int(round(float(budget))))) for budget in spaced]

    spans = [(lo, hi, eta) for lo, hi in ((500, 5000), (1000, 9000), (1000, 8000), (100, 10000), (300, 5000)) for eta in (2, 3, 4)]
    assert _run_figure_program("data.map(([lo, hi, e]) => V.hyperbandBudgets(lo, hi, e))", spans) == [
        backend_budgets(*span) for span in spans
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
def test_ccmo_generation_figure_agrees_with_vamos() -> None:
    from vamos.engine.algorithm.ccmo.selection import (
        ccmo_environmental_selection,
        ccmo_fitness,
        ccmo_tournament,
        constraint_first_dominance,
    )
    from vamos.engine.algorithm.spea2.helpers import strength_raw_fitness

    seeds = [37454, 1, 2, 3, 97, 12118]  # 37454 is the lesson's example
    generations = _run_node(
        f"process.stdout.write(JSON.stringify({json.dumps(seeds)}.map((seed) => globalThis.VamosFoundations.ccmoGeneration(seed, 6))));"
    )

    for g in generations:
        children = g["children"]
        assert [c["population"] for c in children] == [1, 1, 1, 2, 2, 2]  # population 1's offspring first
        for k, pop in enumerate(g["pops"]):
            F = np.array([p["f"] for p in pop])
            cv = np.array([p["cv"] for p in pop]) if k == 0 else np.zeros(len(pop))
            fitness, _ = ccmo_fitness(F, cv)
            np.testing.assert_allclose(g["fitness"][k], fitness, rtol=1e-12)
            # The tournaments, replayed with their draws, and the pairs: the i-th winner of each half.
            mating = g["mating"][k]
            winners = ccmo_tournament(fitness, 6, _Draws(np.array([t["draws"] for t in mating["tournaments"]])))
            assert [pop[i]["id"] for i in winners] == [t["winner"] for t in mating["tournaments"]]
            assert mating["pairs"] == [[pop[winners[i]]["id"], pop[winners[3 + i]]["id"]] for i in range(3)]
            assert [c["parents"] for c in children[3 * k : 3 * k + 3]] == mating["pairs"]
        # Each population selects from itself and every child, the violation first in population 1 only.
        for k, sel in enumerate(g["selections"]):
            candidates = sel["candidates"]
            assert [c["id"] for c in candidates] == [p["id"] for p in g["pops"][k]] + [c["id"] for c in children]
            F = np.array([c["f"] for c in candidates])
            cv = np.array([c["cv"] for c in candidates]) if k == 0 else np.zeros(len(candidates))
            fitness, _ = ccmo_fitness(F, cv)
            np.testing.assert_allclose(sel["fitness"], fitness, rtol=1e-12)
            assert sel["R"] == strength_raw_fitness(constraint_first_dominance(F, cv)).tolist()
            order, _ = ccmo_environmental_selection(F, cv, 6)
            assert sel["kept"] == [candidates[i]["id"] for i in order]  # stored by increasing fitness

    # The numbers that the lesson quotes for its example.
    example = generations[0]
    ties = [(t["a"], t["b"], t["winner"]) for t in example["mating"][1]["tournaments"] if t["tie"] and t["a"] != t["b"]]
    assert ties == [(8, 10, 8)]
    assert round(example["fitness"][1][1], 3) == round(example["fitness"][1][3], 3) == 0.422  # members 8 and 10
    assert sum(t["a"] == t["b"] for t in example["mating"][1]["tournaments"]) == 3
    first, second = example["selections"]
    fitness_of = {c["id"]: v for c, v in zip(first["candidates"], first["fitness"])}
    assert sorted(i for i, v in fitness_of.items() if v < 1) == [6, 13, 16, 17, 18]
    assert first["kept"] == [17, 13, 6, 16, 18, 5] and round(fitness_of[5], 2) == 7.43
    assert all(c["cv"] > 0 for c in first["candidates"][:4])  # parents 1 to 4, all removed, are infeasible
    assert second["rule"] == "truncate" and second["below"] == 7
    assert sorted(second["kept"]) == [8, 9, 10, 12, 14, 17]
    by_id = {c["id"]: c for c in second["candidates"]}
    assert round(float(np.linalg.norm(np.subtract(by_id[18]["f"], by_id[10]["f"]))), 2) == 0.06
    assert by_id[14]["cv"] > 0 and 14 not in first["kept"]


@pytest.mark.skipif(shutil.which("node") is None, reason="the figures' script runs under Node.js")
def test_ctaea_generation_figure_agrees_with_vamos() -> None:
    from vamos.engine.algorithm.ctaea.archives import (
        associate,
        convergence_archive_update,
        diversity_archive_update,
        front_ranks,
        tchebycheff_in_own_subregion,
        worst_of_the_most_crowded_subregion,
    )
    from vamos.engine.algorithm.ctaea.mating import mating_proportions, tournament_winners
    from vamos.engine.algorithm.ctaea.weights import ctaea_weight_vectors

    seeds = [196524, 1, 2, 3, 23, 57983]  # 196524 is the lesson's example; 2 and 23 have fewer and exactly N feasible candidates
    runs = _run_node(
        f"process.stdout.write(JSON.stringify({json.dumps(seeds)}.map((seed) => globalThis.VamosFoundations.ctaeaGeneration(seed, 6, 2))));"
    )
    W = ctaea_weight_vectors(6, 2)
    cases = set()
    for run in runs:
        assert np.array_equal(np.array(run["W"]), W)
        solutions = {s["id"]: s for s in run["solutions"]}

        def F(ids: list[int], solutions: dict[int, Any] = solutions) -> np.ndarray:
            return np.array([solutions[i]["f"] for i in ids]).reshape(-1, 2)

        def cv(ids: list[int], solutions: dict[int, Any] = solutions) -> np.ndarray:
            return np.array([solutions[i]["cv"] for i in ids])

        assert all(s["region"] == int(associate(F([s["id"]]), W)[0]) for s in run["solutions"])
        # Both archives after each update, in VAMOS' order: the initial population, then each generation's children.
        CA: list[int] = []
        DA: list[int] = []
        for history in run["history"]:
            candidates = CA + history["offspring"]
            CA = [candidates[i] for i in convergence_archive_update(F(candidates), cv(candidates), W)]
            candidates = DA + history["offspring"]
            DA = [candidates[i] for i in diversity_archive_update(F(CA), F(candidates), W)]
            assert (CA, DA) == (history["CA"], history["DA"])
        for rec in run["records"]:
            # Restricted mating selection: the shares, then the tournaments replayed with their draws.
            shares = mating_proportions(F(rec["CA"]), F(rec["DA"]))
            assert [rec["shares"][key] for key in ("rhoC", "rhoD", "PC")] == [
                shares.convergence_share,
                shares.diversity_share,
                shares.non_dominated_share_of_convergence_archive,
            ]
            assert (rec["firstFrom"] == "CA") == shares.first_parent_from_convergence_archive
            archives = {"first": rec[rec["firstFrom"]], "caGames": rec["CA"], "daGames": rec["DA"]}
            for name, archive in archives.items():
                winners = tournament_winners(F(archive), cv(archive) == 0.0, np.array([t["draws"] for t in rec[name]]))
                assert [archive[i] for i in winners] == [t["winner"] for t in rec[name]]
            assert rec["fromCA"] == [u < shares.non_dominated_share_of_convergence_archive for u in rec["uniforms"]]
            second = [(rec["caGames"] if c else rec["daGames"])[i]["winner"] for i, c in enumerate(rec["fromCA"])]
            assert [solutions[c]["parents"] for c in rec["children"]] == [[s, t["winner"]] for s, t in zip(second, rec["first"])]
            # The CA update: its case, each removal from the most crowded subregion, or the ranking of the infeasible candidates.
            ca = rec["caUpdate"]
            n_feasible = len(ca["feasible"])
            cases.add("more" if n_feasible > 6 else "fewer" if n_feasible < 6 else "exact")
            for removal in ca["removals"]:
                rows = removal["rows"]
                assert rows[worst_of_the_most_crowded_subregion(F(rows), W)] == removal["removed"]
                low, high = F(rows).min(axis=0), F(rows).max(axis=0)
                regions = associate((F(rows) - low) / (high - low), W)
                assert removal["region"] == regions.tolist() and removal["counts"] == np.bincount(regions, minlength=6).tolist()
                assert removal["crowded"] == int(np.argmax(removal["counts"]))
                np.testing.assert_allclose(removal["tch"], tchebycheff_in_own_subregion(F(removal["closest"]), W), rtol=1e-12)
            if ca["infeasible"]:
                rows = ca["infeasible"]["rows"]
                tch = tchebycheff_in_own_subregion(F(rows), W)
                np.testing.assert_allclose(ca["infeasible"]["tch"], tch, rtol=1e-12)
                ranks = front_ranks(np.column_stack([cv(rows), tch]))
                assert ca["infeasible"]["fronts"] == [[r for r, k in zip(rows, ranks) if k == level] for level in range(ranks.max() + 1)]
            # The DA update counts the new CA's members in each subregion and picks, subregion by subregion, the best candidate.
            da = rec["daUpdate"]
            assert da["caCounts"] == np.bincount(associate(F(ca["kept"]), W), minlength=6).tolist()
            assert [t["chosen"] for t in da["trace"]] == da["kept"]
            for t in da["trace"]:
                np.testing.assert_allclose(t["tch"], tchebycheff_in_own_subregion(F(t["nd"]), W), rtol=1e-12)
    assert cases == {"more", "fewer", "exact"}

    # The numbers that the lesson quotes for its example.
    example = runs[0]["records"][1]
    assert (sorted(example["CA"]), sorted(example["DA"])) == ([4, 5, 7, 8, 9, 12], [1, 2, 3, 6, 10, 12])
    assert (example["shares"]["inFirst"], example["shares"]["inSecond"], example["shares"]["inCA"]) == (0, 3, 5)
    assert example["firstFrom"] == "DA"
    assert [t["reason"] for t in example["first"]].count("third") == 4
    games = {(t["a"], t["b"]): (t["winner"], t["reason"]) for t in example["second"]}
    assert games[(7, 4)] == (4, "dominates")
    ca = example["caUpdate"]
    assert len(ca["feasible"]) == 8 and ca["fronts"] == [[4, 5, 8, 9, 12], [15, 18], [7]]
    (removal,) = ca["removals"]
    assert removal["crowded"] == 0 and sorted(removal["closest"]) == [5, 15] and removal["removed"] == 15
    assert dict(zip(removal["closest"], np.round(removal["tch"], 3))) == {5: 0.105, 15: 0.315}
    da = example["daUpdate"]
    assert da["caCounts"] == [0, 2, 0, 3, 1, 0]
    assert [(t["iteration"], t["region"], t["chosen"]) for t in da["trace"]] == [
        (1, 2, 16),
        (2, 2, 3),
        (2, 2, 6),
        (2, 4, 2),
        (3, 1, 17),
        (3, 4, 13),
    ]
    assert sorted(ca["kept"]) == [4, 5, 8, 9, 12, 18]
    assert all(runs[0]["solutions"][i - 1]["cv"] > 0 for i in da["kept"])


@pytest.mark.skipif(shutil.which("node") is None, reason="the figures' script runs under Node.js")
def test_benchmark_figures_agree_with_vamos() -> None:
    """The Benchmarks figures: their fronts match VAMOS' packaged fronts, and their maps VAMOS' problems."""
    from scipy.spatial import cKDTree

    from vamos import make_problem_selection
    from vamos.foundation.problem.resolver import resolve_reference_front_path
    from vamos.problems import DTLZ4
    from vamos.resources import reference_front_path

    rng = np.random.default_rng(20261122)
    positions = rng.random((40, 4))
    positions[0], positions[1] = 0.0, 1.0
    ctp_designs = {}
    for key in ("ctp1", "ctp2", "ctp3", "ctp4", "ctp5", "ctp6", "ctp7"):
        problem = make_problem_selection(key).instantiate()
        X = rng.uniform(problem.xl, problem.xu, size=(200, problem.n_var))
        X[:100, 1:] = rng.normal(0.0, 0.05, size=(100, problem.n_var - 1))  # g close to 1, near the fronts
        out: dict[str, np.ndarray] = {}
        problem.evaluate(X, out)
        ctp_designs[key] = (out["F"], np.maximum(out["G"], 0.0).sum(axis=1))
    program = f"""
require({json.dumps(str(SCRIPT))});
const V = globalThis.VamosFoundations;
const D = JSON.parse(require("fs").readFileSync(0, "utf8"));
const fronts = {{}};
for (const key of Object.keys(V.BENCHMARK_FRONTS)) fronts[key] = V.benchmarkFront(key, 4001);
const ctpFronts = {{}};
const violations = {{}};
for (const key of Object.keys(D.ctp)) {{
  ctpFronts[key] = V.ctpFront(key, 1001);
  violations[key] = D.ctp[key].map((f) => V.ctpTotalViolation(key, f));
}}
const bias = {{
  dtlz4: Object.fromEntries([1, 5, 100].map((a) => [a, D.u.map((u) => V.biasPoint("dtlz4", [u[0]], a))])),
  zdt6: D.u.map((u) => V.biasPoint("zdt6", [u[0]])),
  wfg1: D.u.map((u) => V.biasPoint("wfg1", u)),
}};
process.stdout.write(JSON.stringify({{ fronts, ctpFronts, violations, bias }}));
"""
    payload = json.dumps({"u": positions.tolist(), "ctp": {key: F.tolist() for key, (F, _) in ctp_designs.items()}})
    completed = subprocess.run(["node", "-e", program], input=payload, capture_output=True, text=True, check=True, timeout=120)
    js = json.loads(completed.stdout)

    def assert_close_sets(figure: np.ndarray, reference: np.ndarray, label: str) -> None:
        # Every point of each set lies within 1% of the reference front's range of a point of the other.
        scale = reference.max(axis=0) - reference.min(axis=0)
        a, b = figure / scale, reference / scale
        assert cKDTree(b).query(a)[0].max() < 0.01, f"{label}: a figure point is off the reference front"
        assert cKDTree(a).query(b)[0].max() < 0.01, f"{label}: the figure misses part of the reference front"

    for key, points in js["fronts"].items():
        path = resolve_reference_front_path(key, None, n_obj=2)
        assert path is not None, key
        assert_close_sets(np.asarray(points), np.loadtxt(path, delimiter=","), key)
    for key, points in js["ctpFronts"].items():
        assert_close_sets(np.asarray(points), np.loadtxt(reference_front_path(key), delimiter=","), key)
        np.testing.assert_allclose(js["violations"][key], ctp_designs[key][1], rtol=1e-9, atol=1e-9, err_msg=key)

    for alpha, points in js["bias"]["dtlz4"].items():
        X = np.full((len(positions), 11), 0.5)
        X[:, 0] = positions[:, 0]
        out = {}
        DTLZ4(n_var=11, n_obj=2, alpha=float(alpha)).evaluate(X, out)
        np.testing.assert_allclose(points, out["F"], atol=1e-12, err_msg=f"DTLZ4, alpha = {alpha}")
    zdt6 = make_problem_selection("zdt6").instantiate()
    X = np.zeros((len(positions), zdt6.n_var))
    X[:, 0] = positions[:, 0]
    out = {}
    zdt6.evaluate(X, out)
    np.testing.assert_allclose(js["bias"]["zdt6"], out["F"], atol=1e-12)
    # One distance variable, so that its optimum 0.35 * x_u is exactly 0.35 after normalization.
    wfg1 = make_problem_selection("wfg1", n_var=5, n_obj=2).instantiate()
    xu = np.asarray(wfg1.xu, dtype=float)
    X = np.tile(0.35 * xu, (len(positions), 1))
    assert X[0, 4] / xu[4] == 0.35
    X[:, :4] = positions * xu[:4]
    out = {}
    wfg1.evaluate(X, out)
    np.testing.assert_allclose(js["bias"]["wfg1"], out["F"], atol=1e-12)


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


@pytest.mark.skipif(shutil.which("node") is None, reason="the figures' script runs under Node.js")
def test_nsga2_generation_figure_agrees_with_vamos() -> None:
    from vamos import select_survivors

    seeds = [4679, 1, 2, 3, 97, 12367]  # 4679 is the lesson's example
    program = f"""
require({json.dumps(str(SCRIPT))});
const V = globalThis.VamosFoundations;
const generations = {json.dumps(seeds)}.map((seed) => V.nsga2Generation(seed, 6));
process.stdout.write(JSON.stringify(generations, (key, value) => (value === Infinity ? "inf" : value)));
"""
    completed = subprocess.run(["node", "-e", program], capture_output=True, text=True, check=True, timeout=60)
    generations = json.loads(completed.stdout)

    # Fronts, crowding distances and survivors of the merged population are VAMOS's.
    for g in generations:
        F = np.array([member["f"] for member in g["merged"]])
        fronts, _ = _fast_non_dominated_sort(F)
        assert [sorted(front) for front in g["fronts"]] == [sorted(int(i) for i in front) for front in fronts]
        crowding = np.array([np.inf if d == "inf" else d for d in g["mergedCrowding"]], dtype=float)
        np.testing.assert_allclose(crowding, _compute_crowding(F, fronts), rtol=1e-12)
        assert sorted(g["keep"]) == sorted(int(i) for i in select_survivors(F, survivor_count=6).selected_indices)
        # The whole fronts come first and are kept; the next front, if any, has more members than places.
        assert g["whole"] == g["fronts"][: len(g["whole"])]
        assert sum(len(front) for front in g["whole"]) + g["places"] == 6
        assert {i for front in g["whole"] for i in front} <= set(g["keep"])
        if g["split"] is not None:
            assert g["split"] == g["fronts"][len(g["whole"])] and len(g["split"]) > g["places"]

    # The numbers that the lesson quotes for its example.
    example = generations[0]
    ident = [member["id"] for member in example["merged"]]
    games = {(t["a"], t["b"]): (t["winner"], t["reason"]) for t in example["tournaments"]}
    assert games[(1, 4)] == (1, "crowding") and games[(2, 5)] == (2, "coin")
    assert [sorted(ident[i] for i in front) for front in example["whole"]] == [[1, 9]]
    assert sorted(ident[i] for i in example["split"]) == [2, 3, 4, 5, 8, 11] and example["places"] == 4
    distances = {ident[i]: example["mergedCrowding"][i] for i in example["split"]}
    assert distances[3] == distances[5] == "inf"
    assert {k: round(distances[k], 2) for k in (4, 11, 8, 2)} == {4: 1.05, 11: 0.92, 8: 0.81, 2: 0.55}
    assert sorted(ident[i] for i in example["keep"]) == [1, 3, 4, 5, 9, 11]
