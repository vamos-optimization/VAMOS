"""Regenerate ``tests/engine/data/subproblem_update_jmetal_decisions.json`` with jMetal's criteria.

Each case is a sequence of steps of one subproblem update criterion of jMetal (package
``component.catalogue.ea.replacement.subproblemupdate``). A step gives the population and a new
solution, which ``update`` receives, and candidates: the constraint values of a current solution and
the aggregation values of the new and current solutions, which ``replaces`` compares. The fixture
holds every decision, and epsilon or the threshold after each step.
``tests/engine/test_subproblem_update_criteria.py`` checks VAMOS's criteria against them.

The constraint values are small multiples of 0.25 and the aggregation values multiples of 0.1, so
that violations, measures and values tie, and epsilon often equals a violation. The populations change
by one row per step, as MOEA/D's do; the criteria only read them. A case stores its first population,
and each step the new solution, the rows of the candidates and the row that the new solution then
takes (:func:`populations` rebuilds the population of each step).

Run from a checkout::

    python tools/subproblem_update_reference_decisions.py --classpath "<classes and jars>"

``--classpath`` lists, separated by the platform's path separator, the criteria compiled from
jMetal's ``main`` at commit ``JMETAL_COMMIT`` (``component/catalogue/ea/replacement/subproblemupdate``)
and ``util/ConstraintHandling.java`` at that commit first, then ``jmetal-core`` with its dependencies.
``tools/jmetal_values/SubproblemUpdateDecisions.java`` runs with Java 17 or later. Constraint values are
stored in the VAMOS convention, ``g <= 0`` feasible, and negated for jMetal.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "engine" / "data" / "subproblem_update_jmetal_decisions.json"
JAVA_HELPER = ROOT / "tools" / "jmetal_values" / "SubproblemUpdateDecisions.java"
JMETAL_COMMIT = "46fb0a61f7"
SEED = 20261113
LEVELS = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0])


def _rows(rng: np.random.Generator, rows: int, k: int, feasible: float) -> np.ndarray:
    G = rng.choice(LEVELS, size=(rows, k))
    keep = rng.random(rows) < feasible
    G[keep] = -np.abs(G[keep])
    return G


def _case(
    rng: np.random.Generator,
    criterion: str,
    n: int,
    k: int,
    steps: int,
    feasible: tuple[float, float],
    *,
    tao: float = 0.0,
    threshold: float = 0.0,
    tc: int = 0,
    candidates: int = 4,
) -> dict[str, Any]:
    """A case whose population goes from ``feasible[0]`` to ``feasible[1]`` feasible rows, step by step."""
    population = _rows(rng, n, k, feasible[0])
    case_steps: list[dict[str, Any]] = []
    for step in range(steps):
        share = feasible[0] + (feasible[1] - feasible[0]) * step / max(1, steps - 1)
        case_steps.append(
            {
                "child": _rows(rng, 1, k, share)[0].tolist(),
                "rows": rng.choice(n, size=min(candidates, n), replace=False).tolist(),
                "child_values": (rng.integers(1, 6, size=min(candidates, n)) / 10).tolist(),
                "values": (rng.integers(1, 6, size=min(candidates, n)) / 10).tolist(),
                "replaced": int(rng.integers(n)),
            }
        )
    return {
        "criterion": criterion,
        "params": {"tao": tao, "feasibility_ratio_threshold": threshold, "tc": tc},
        "n": n,
        "k": k,
        "population": population.tolist(),
        "steps": case_steps,
    }


def populations(case: dict[str, Any]) -> list[np.ndarray]:
    """The population of each step of a case: the previous one, with the previous new solution in its row."""
    population = np.asarray(case["population"], dtype=float)
    result = []
    for step in case["steps"]:
        result.append(population)
        population = population.copy()
        population[step["replaced"]] = step["child"]
    return result


def _cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    return [
        # Epsilon shrinks while few rows are feasible, then reaches 0 at generation tc.
        _case(rng, "improved_epsilon", 10, 2, 40, (0.1, 0.3), tao=0.1, threshold=0.95, tc=3),
        # Most rows become feasible: epsilon grows to the largest violation seen times 1 + tao.
        _case(rng, "improved_epsilon", 10, 2, 40, (0.4, 0.9), tao=0.2, threshold=0.5, tc=10),
        # 25 rows: the initial epsilon is the third most violated row's violation.
        _case(rng, "improved_epsilon", 25, 1, 75, (0.2, 0.6), tao=0.05, threshold=0.6, tc=5),
        # Fewer than 20 rows: the initial epsilon is the second most violated row's violation.
        _case(rng, "improved_epsilon", 6, 3, 24, (0.3, 0.5), tao=0.3, threshold=0.4, tc=2, candidates=6),
        # Every row feasible: epsilon and the largest violation stay 0.
        _case(rng, "improved_epsilon", 8, 2, 16, (1.0, 1.0), tao=0.1, threshold=0.95, tc=4),
        _case(rng, "violation_threshold", 8, 2, 32, (0.2, 0.7)),
        _case(rng, "violation_threshold", 12, 3, 36, (0.5, 0.5), candidates=6),
        # Every row feasible: the threshold is 0.
        _case(rng, "violation_threshold", 6, 2, 12, (1.0, 1.0)),
        # Every row infeasible.
        _case(rng, "violation_threshold", 6, 1, 18, (0.0, 0.0)),
    ]


def _java_input(cases: list[dict[str, Any]]) -> str:
    def numbers(values: Any) -> str:
        return " ".join(repr(float(value)) for value in np.ravel(values))

    lines = [str(len(cases))]
    for case in cases:
        params = case["params"]
        lines.append(
            f"{case['criterion']} {params['tao']!r} {params['feasibility_ratio_threshold']!r} {params['tc']} {case['n']} {case['k']}"
        )
        lines.append(str(len(case["steps"])))
        for step, population in zip(case["steps"], populations(case)):
            lines.append(numbers(population))
            lines.append(numbers(step["child"]))
            lines.append(str(len(step["rows"])))
            for row, child_value, value in zip(step["rows"], step["child_values"], step["values"]):
                lines.append(f"{numbers(population[row])} {child_value!r} {value!r}")
    return "\n".join(lines) + "\n"


def _read_output(cases: list[dict[str, Any]], output: str) -> None:
    lines = iter(output.splitlines())
    for case in cases:
        for step in case["steps"]:
            decisions = next(lines).split()
            assert decisions[0] == "replaces", decisions
            step["replaces"] = [value == "1" for value in decisions[1:]]
            state = next(lines).split()
            assert state[0] == "state", state
            value = float(state[1])
            step["state"] = None if np.isnan(value) else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the fixture of jMetal's subproblem update decisions.")
    parser.add_argument("--classpath", required=True)
    parser.add_argument("--java", default="java")
    args = parser.parse_args()
    cases = _cases(np.random.default_rng(SEED))
    result = subprocess.run(
        [args.java, "-cp", args.classpath, str(JAVA_HELPER)],
        input=_java_input(cases),
        capture_output=True,
        text=True,
        check=True,
    )
    _read_output(cases, result.stdout)
    document = {
        "source": f"jMetal main at {JMETAL_COMMIT}, package component.catalogue.ea.replacement.subproblemupdate",
        "generator": "tools/subproblem_update_reference_decisions.py",
        "seed": SEED,
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(document, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"{OUTPUT.relative_to(ROOT)}: {len(cases)} cases, {sum(len(case['steps']) for case in cases)} steps")


if __name__ == "__main__":
    main()
