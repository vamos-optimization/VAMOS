"""Regenerate ``tests/engine/data/constraint_handling_jmetal_ranks.json`` with jMetal's techniques.

Each case is a fixed population that a constraint-handling technique of jMetal (package
``util.constrainthandling``) ranks, after the updates the case lists (a population and the run's
progress each). The fixture holds the ranks, and epsilon and the push phase where the technique has
them. ``tests/engine/test_constraint_handling_techniques.py`` checks VAMOS's techniques against them.
Stochastic ranking draws its uniforms from ``numpy.random.default_rng(uniform_seed).random(n * (n - 1))``,
which jMetal replays in order.

Run from a checkout, with VAMOS importable::

    PYTHONPATH=src python tools/constraint_handling_reference_ranks.py --classpath "<classes and jars>"

``--classpath`` lists, separated by the platform's path separator, jMetal's ``jmetal-core`` with
commons-lang3 and commons-math3, the techniques compiled from jMetal's ``main`` at commit
``JMETAL_COMMIT`` (``util/constrainthandling`` and ``util/ConstraintHandling.java``) first;
``tools/jmetal_values/ChtRanks.java`` runs with Java 17 or later. Constraint values are stored in the
VAMOS convention, ``g <= 0`` feasible, and negated for jMetal.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "engine" / "data" / "constraint_handling_jmetal_ranks.json"
JAVA_HELPER = ROOT / "tools" / "jmetal_values" / "ChtRanks.java"
JMETAL_COMMIT = "46fb0a61f7"
SEED = 20261112

EPSILON = {"initial_quantile": 0.5, "exponent": 2.0, "decay_fraction": 0.8}
PUSH_PULL = {
    "change_threshold": 1e-3,
    "window_fraction": 0.05,
    "decay_fraction": 0.8,
    "exponent": 2.0,
    "feasibility_threshold": 0.95,
    "reduction_rate": 0.1,
}


def _population(
    rng: np.random.Generator, n: int, n_obj: int, n_constraints: int, *, feasible: float = 0.5, integer: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """``n`` rows, a fraction ``feasible`` of them feasible, with two duplicated objective vectors."""
    F = rng.random((n, n_obj)).round(4)
    F[n - 1] = F[0]
    F[n - 2] = F[1]
    is_feasible = rng.random(n) < feasible
    if integer:
        # Small integers: many rows share their violation, or their number of violated constraints.
        G = rng.integers(-2, 3, size=(n, n_constraints)).astype(float)
    else:
        G = rng.normal(size=(n, n_constraints)).round(4)
    G[is_feasible] = -np.abs(G[is_feasible])
    for i in np.flatnonzero(~is_feasible):
        if np.all(G[i] <= 0.0):
            G[i, rng.integers(n_constraints)] = 1.0 if integer else float(np.abs(G[i]).max().round(4)) + 0.01
    return F, G


def _cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def add(technique: str, params: dict[str, float], population: tuple[np.ndarray, np.ndarray], **extra: Any) -> None:
        F, G = population
        case: dict[str, Any] = {"technique": technique, "params": params, "updates": [], "F": F.tolist(), "G": G.tolist()}
        case.update(extra)
        cases.append(case)

    def update(progress: float, population: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
        return {"progress": progress, "F": population[0].tolist(), "G": population[1].tolist()}

    for name in ("cdp", "violated_constraints"):
        add(name, {}, _population(rng, 30, 2, 3))
        add(name, {}, _population(rng, 30, 2, 3, integer=True))
        add(name, {}, _population(rng, 20, 3, 2, feasible=0.0))
        add(name, {}, _population(rng, 20, 2, 1, feasible=1.0))

    # Mostly infeasible populations, so that epsilon starts above 0.
    first = _population(rng, 30, 2, 2, feasible=0.2)
    later = _population(rng, 30, 2, 2, feasible=0.3)
    add("epsilon_constrained", EPSILON, first, updates=[update(0.0, first)])
    add("epsilon_constrained", EPSILON, later, updates=[update(0.0, first), update(0.4, later)])
    add("epsilon_constrained", EPSILON, later, updates=[update(0.0, first), update(0.9, later)])
    ties = _population(rng, 30, 2, 3, feasible=0.1, integer=True)
    add("epsilon_constrained", {**EPSILON, "initial_quantile": 0.2}, ties, updates=[update(0.0, ties), update(0.1, ties)])
    add("epsilon_constrained", {**EPSILON, "initial_quantile": 0.0}, ties, updates=[update(0.0, ties)])

    still = _population(rng, 30, 2, 2)
    moving = [_population(rng, 30, 2, 2) for _ in range(3)]
    add("push_pull", PUSH_PULL, moving[1], updates=[update(0.0, moving[0]), update(0.02, moving[1])])
    settles = [update(round(0.02 * i, 2), still) for i in range(4)]
    add("push_pull", PUSH_PULL, still, updates=settles)
    feasible = _population(rng, 30, 2, 2, feasible=1.0)
    add("push_pull", PUSH_PULL, feasible, updates=[*settles, update(0.3, feasible)])
    add("push_pull", PUSH_PULL, still, updates=[*settles, update(0.3, still), update(0.32, still)])
    sliding = [update(round(0.02 * i, 2), moving[i]) for i in range(3)] + [update(round(0.02 * i, 2), still) for i in range(3, 9)]
    add("push_pull", PUSH_PULL, still, updates=sliding)
    add("push_pull", PUSH_PULL, still, updates=[update(0.0, moving[0]), update(0.85, still)])

    for probability, population in (
        (0.45, _population(rng, 30, 2, 2)),
        (0.0, _population(rng, 30, 2, 2)),
        (1.0, _population(rng, 30, 2, 2)),
        (0.45, _population(rng, 25, 3, 2, feasible=0.0)),
        (0.45, _population(rng, 30, 2, 3, integer=True)),
        (0.45, _population(rng, 20, 2, 1, feasible=1.0)),
    ):
        add("stochastic_ranking", {"probability": probability}, population, uniform_seed=int(rng.integers(2**31)))

    add("self_adaptive_penalty", {}, _population(rng, 30, 2, 3))
    add("self_adaptive_penalty", {}, _population(rng, 30, 3, 2))
    add("self_adaptive_penalty", {}, _population(rng, 20, 2, 2, feasible=0.0))
    add("self_adaptive_penalty", {}, _population(rng, 20, 2, 2, feasible=1.0))
    never = _population(rng, 30, 2, 3)
    never[1][:, 2] = -np.abs(never[1][:, 2])
    add("self_adaptive_penalty", {}, never)
    return cases


def _uniforms(case: dict[str, Any]) -> np.ndarray:
    if "uniform_seed" not in case:
        return np.empty(0)
    n = len(case["F"])
    return np.random.default_rng(case["uniform_seed"]).random(n * (n - 1))


def _rows(F: list[list[float]], G: list[list[float]]) -> str:
    return "\n".join(" ".join(repr(float(v)) for v in (*f, *g)) for f, g in zip(F, G))


def _jmetal(classpath: str, cases: list[dict[str, Any]]) -> list[tuple[list[int], float, str]]:
    parts = [str(len(cases))]
    for case in cases:
        params = list(case["params"].values())
        n_obj, n_constraints = len(case["F"][0]), len(case["G"][0])
        parts.append(f"{case['technique']} {len(params)} {' '.join(repr(float(p)) for p in params)} {n_obj} {n_constraints}")
        parts.append(str(len(case["updates"])))
        for step in case["updates"]:
            parts.append(f"{step['progress']!r} {len(step['F'])}")
            parts.append(_rows(step["F"], step["G"]))
        parts.append(str(len(case["F"])))
        parts.append(_rows(case["F"], case["G"]))
        uniforms = _uniforms(case)
        parts.append(" ".join([str(uniforms.size), *(repr(float(u)) for u in uniforms)]))
    completed = subprocess.run(
        ["java", "-cp", classpath, str(JAVA_HELPER)], input="\n".join(parts) + "\n", capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ChtRanks.java failed:\n{completed.stderr}")
    lines = str(completed.stdout).splitlines()
    results = []
    for ranks_line, state_line in zip(lines[0::2], lines[1::2]):
        _, epsilon, pushing, _ = state_line.split()
        results.append(([int(r) for r in ranks_line.split()[1:]], float(epsilon), pushing))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classpath", required=True, help="jMetal's techniques at JMETAL_COMMIT, jmetal-core and its dependencies")
    args = parser.parse_args()
    cases = _cases(np.random.default_rng(SEED))
    for case, (ranks, epsilon, pushing) in zip(cases, _jmetal(args.classpath, cases), strict=True):
        case["ranks"] = ranks
        if case["technique"] in ("epsilon_constrained", "push_pull"):
            case["epsilon"] = epsilon
        if case["technique"] == "push_pull":
            case["pushing"] = pushing == "true"
    payload = {
        "source": f"jMetal main {JMETAL_COMMIT}, package org.uma.jmetal.util.constrainthandling",
        "generator": "tools/constraint_handling_reference_ranks.py",
        "cases": cases,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(cases)} cases written to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
