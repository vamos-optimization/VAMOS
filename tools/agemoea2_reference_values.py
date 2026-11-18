"""Regenerate ``tests/engine/data/agemoea2_jmetal_values.json`` with jMetal's AGE-MOEA-II selection.

Three kinds of case, read by ``tools/jmetal_values/AgeMoea2Values.java``:

- ``geometry``: a normalized front and its extreme rows; jMetal's ``computeGeometry`` gives the
  exponent ``p`` (Newton-Raphson from the non-extreme row nearest to the origin);
- ``distances``: a normalized front and ``p``; jMetal's ``pairwiseDistances`` gives the geodesic
  distances, through the projected midpoints;
- ``survival``: a first front in raw objectives; jMetal's ``computeSurvivalScore`` gives the survival
  scores of AGE-MOEA and of AGE-MOEA-II (infinite for the extreme rows, stored as ``null``). VAMOS
  shares AGE-MOEA's other steps, whose extreme rows and tie-breaking follow PlatEMO and pymoo rather
  than jMetal, so the test compares these scores only where those steps agree (``q`` records the
  sphere: with ``q = 1`` a two-objective front is a segment, where the greedy selection ties).

The fronts are points of ``L_q`` spheres, ``u / ||u||_q`` for positive ``u``, with ``q`` from 0.4
(convex) to 6 (near a hypercube), so that the exponents span the clamp of 0.1 and the cap of 20; the
survival fronts are scaled and shifted per objective, and the geometry and distance fronts include
coordinates that are 0 or above 1. Every survival front has more rows than objectives, as jMetal
scores only those (``computeSurvivalScore`` returns before the geometry otherwise).
``tests/engine/test_agemoea_ii.py`` checks VAMOS's survival against them.

Run from a checkout::

    python tools/agemoea2_reference_values.py --classpath "<classes and jars>"

``--classpath`` lists, separated by the platform's path separator, the classes
``AGEMOEAEnvironmentalSelection``, ``AGEMOEA2EnvironmentalSelection`` and ``SurvivalScoreComparator``
compiled from jMetal's ``main`` at commit ``JMETAL_COMMIT``
(``jmetal-component/.../ea/replacement/impl/agemoea``) first, then ``jmetal-core`` and
``jmetal-component`` with their dependencies. ``tools/jmetal_values/AgeMoea2Values.java`` runs with Java
21 or later (source-file mode).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "engine" / "data" / "agemoea2_jmetal_values.json"
JAVA_HELPER = ROOT / "tools" / "jmetal_values" / "AgeMoea2Values.java"
JMETAL_COMMIT = "46fb0a61f7"
SEED = 20261118
EXPONENTS = (0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 6.0)


def _sphere(rng: np.random.Generator, m: int, n: int, q: float) -> np.ndarray:
    """``m`` points of the positive orthant of the unit ``L_q`` sphere."""
    U = rng.uniform(0.05, 1.0, size=(m, n))
    return np.asarray(U / np.sum(U**q, axis=1, keepdims=True) ** (1.0 / q), dtype=float)


def _extremes(front: np.ndarray) -> list[int]:
    """The row nearest to each axis, as AGE-MOEA picks its extreme points (distinct rows here)."""
    chosen: list[int] = []
    for j in range(front.shape[1]):
        others = np.delete(front, j, axis=1)
        order = np.argsort(np.sum(others**2, axis=1), kind="stable")
        chosen.append(int(next(i for i in order if i not in chosen)))
    return chosen


def _cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for n in (2, 3, 5):
        for q in EXPONENTS:
            for m in (8, 25):
                front = _sphere(rng, m, n, q)
                cases.append({"kind": "geometry", "q": q, "front": front.tolist(), "extremes": _extremes(front)})
    # A row with a zero coordinate (the derivative skips it) and rows beyond the unit box.
    front = _sphere(rng, 12, 3, 2.0)
    front[5, 1] = 0.0
    front[7] *= 1.4
    cases.append({"kind": "geometry", "front": front.tolist(), "extremes": _extremes(front)})
    # The nearest row on the diagonal of a convex front: p falls below 0.1 or diverges.
    front = _sphere(rng, 10, 3, 0.4)
    front[4] = np.full(3, 0.01)
    cases.append({"kind": "geometry", "front": front.tolist(), "extremes": _extremes(front)})

    for n in (2, 3, 5):
        for p in (0.5, 1.0, 1.7, 2.0, 3.7, 20.0):
            front = _sphere(rng, 9, n, float(rng.choice(EXPONENTS)))
            cases.append({"kind": "distances", "p": p, "front": front.tolist()})
    front = _sphere(rng, 7, 3, 2.0)
    front[2, 0] = 0.0
    cases.append({"kind": "distances", "p": 2.4, "front": front.tolist()})

    for n in (2, 3, 4, 5):
        for q in EXPONENTS:
            for m in (n + 3, 30):
                scale = rng.uniform(0.5, 50.0, size=n)
                shift = rng.uniform(-10.0, 10.0, size=n)
                front = _sphere(rng, m, n, q) * scale + shift
                cases.append({"kind": "survival", "q": q, "front": front.tolist()})
    return cases


def _java_input(cases: list[dict[str, Any]]) -> str:
    lines = [str(len(cases))]
    for case in cases:
        front = np.asarray(case["front"], dtype=float)
        m, n = front.shape
        head = f"{case['kind']} {m} {n}"
        if case["kind"] == "distances":
            head += f" {case['p']!r}"
        lines.append(head)
        lines.extend(" ".join(repr(float(v)) for v in row) for row in front)
        if case["kind"] == "geometry":
            lines.append(f"{len(case['extremes'])} " + " ".join(str(i) for i in case["extremes"]))
    return "\n".join(lines) + "\n"


def _value(text: str) -> float | None:
    value = float(text)
    return None if np.isinf(value) else value


def _read_output(cases: list[dict[str, Any]], stdout: str) -> None:
    lines = stdout.strip().splitlines()
    if len(lines) != len(cases):
        raise RuntimeError(f"expected {len(cases)} lines from the Java helper, got {len(lines)}")
    for case, line in zip(cases, lines, strict=True):
        kind, *values = line.split()
        if kind != case["kind"]:
            raise RuntimeError(f"case kind {case['kind']} answered as {kind}")
        m = len(case["front"])
        if kind == "geometry":
            case["p"] = float(values[0])
        elif kind == "distances":
            case["distances"] = np.asarray([float(v) for v in values]).reshape(m, m).tolist()
        else:
            case["agemoea_scores"] = [_value(v) for v in values[:m]]
            case["agemoea_ii_scores"] = [_value(v) for v in values[m:]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the fixture of jMetal's AGE-MOEA-II selection values.")
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
        "source": f"jMetal main at {JMETAL_COMMIT}, package component.catalogue.ea.replacement.impl.agemoea",
        "generator": "tools/agemoea2_reference_values.py",
        "seed": SEED,
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(document, separators=(",", ":")) + "\n", encoding="utf-8")
    counts = {kind: sum(case["kind"] == kind for case in cases) for kind in ("geometry", "distances", "survival")}
    print(f"{OUTPUT.relative_to(ROOT)}: {len(cases)} cases {counts}")


if __name__ == "__main__":
    main()
