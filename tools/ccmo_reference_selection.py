"""Regenerate ``tests/engine/data/ccmo_jmetal_selection.json`` with jMetal's CCMO.

Each case is a fixed population: objective vectors and a constraint violation per row. The
fixture holds the fitness that jMetal's ``CCMOFitness`` computes with and without the violations,
and, for several population sizes, the rows that ``CCMOEnvironmentalSelection`` selects, in their
order, with their fitness. ``tests/engine/test_ccmo.py`` checks VAMOS's CCMO against them.

Run from a checkout::

    python tools/ccmo_reference_selection.py --jmetal-sources <dir> --classpath "<jars>"

``--jmetal-sources`` is a directory with ``CCMOFitness.java`` and ``CCMOEnvironmentalSelection.java``
of jMetal pull request #521 at commit ``JMETAL_COMMIT``. ``--classpath`` lists, separated by the
platform's path separator, jMetal's ``jmetal-core`` (7.10 or later) with commons-lang3 and
commons-math3. Both classes are package-private, so the tool compiles them with
``tools/jmetal_values/CcmoSelection.java``, which shares their package, and runs it with Java 17
or later (``javac`` and ``java`` on the path). Violations are stored as non-negative numbers, the
VAMOS ``compute_violation`` of a single constraint ``g = violation``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "engine" / "data" / "ccmo_jmetal_selection.json"
JAVA_HELPER = ROOT / "tools" / "jmetal_values" / "CcmoSelection.java"
JAVA_MAIN = "org.uma.jmetal.algorithm.multiobjective.ccmo.CcmoSelection"
JMETAL_SOURCES = ("CCMOFitness.java", "CCMOEnvironmentalSelection.java")
JMETAL_COMMIT = "7c896fcba37a545f32d860f1e0248d75ecb91416"
SEED = 20261113


def _case(F: np.ndarray, violation: np.ndarray, sizes: list[int], label: str) -> dict[str, Any]:
    return {"label": label, "F": F.tolist(), "violation": violation.tolist(), "sizes": sizes}


def _cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    # The five vectors of jMetal's unit tests: a, b and c are non-dominated, b dominates d, all dominate e.
    five = np.array([[1.0, 4.0], [2.0, 2.0], [4.0, 1.0], [3.0, 3.0], [5.0, 5.0]])
    cases.append(_case(five, np.zeros(5), [3, 4], "jmetal-five"))
    cases.append(_case(five, np.array([1.0, 0.0, 0.0, 0.0, 0.0]), [2, 3, 4], "jmetal-five-infeasible-a"))
    cases.append(_case(np.array([[1.0, 1.0], [2.0, 2.0], [10.0, 10.0]]), np.array([2.0, 2.0, 0.5]), [1, 2], "equal-violations"))
    # Evenly spaced points on a line: the truncation meets exact ties in its sorted distances.
    line = np.stack([np.arange(9.0), 8.0 - np.arange(9.0)], axis=1)
    cases.append(_case(line, np.zeros(9), [3, 5, 8], "even-front"))
    # A front with duplicated points, which the truncation removes first.
    duplicated = np.concatenate([line[:5], line[1:4], line[2:3]])
    cases.append(_case(duplicated, np.zeros(duplicated.shape[0]), [4, 6], "duplicated-front"))

    def random_population(n: int, n_obj: int, *, feasible: float, integer_violation: bool, grid: float | None) -> tuple[Any, Any]:
        F = rng.random((n, n_obj))
        F = np.round(F / grid) * grid if grid else F.round(6)
        violation = rng.integers(1, 4, size=n).astype(float) if integer_violation else rng.random(n).round(6) + 1e-3
        violation[rng.random(n) < feasible] = 0.0
        return F, violation

    for n, n_obj, feasible, integer_violation, grid, sizes in (
        (30, 2, 0.5, False, None, [10, 15, 25]),
        (40, 2, 0.0, True, None, [10, 20, 30]),
        (40, 3, 0.3, True, 0.25, [10, 20]),
        (50, 2, 1.0, False, 0.125, [10, 20, 40]),
        (60, 3, 0.5, False, None, [30]),
        (200, 2, 0.5, False, None, [100]),
        (300, 3, 0.3, True, None, [100]),
    ):
        F, violation = random_population(n, n_obj, feasible=feasible, integer_violation=integer_violation, grid=grid)
        cases.append(_case(F, violation, sizes, f"random-{n}x{n_obj}"))

    # A dense non-dominated front, larger than the population: the truncation removes most of it.
    angles = np.sort(rng.random(150)) * np.pi / 2
    front = np.stack([np.cos(angles), np.sin(angles)], axis=1).round(6)
    front_violation = np.where(rng.random(150) < 0.2, 0.5, 0.0)
    cases.append(_case(front, front_violation, [50, 100], "dense-front"))
    return cases


def _jmetal(sources: Path, classpath: str, cases: list[dict[str, Any]]) -> list[str]:
    parts = [str(len(cases))]
    for case in cases:
        F, violation = case["F"], case["violation"]
        parts.append(f"{len(F)} {len(F[0])}")
        parts.extend(" ".join(repr(float(v)) for v in (*row, cv)) for row, cv in zip(F, violation, strict=True))
        parts.append(str(2 * len(case["sizes"])))
        parts.extend(f"{size} {flag}" for size in case["sizes"] for flag in (1, 0))
    with tempfile.TemporaryDirectory() as classes:
        files = [str(sources / name) for name in JMETAL_SOURCES] + [str(JAVA_HELPER)]
        compiled = subprocess.run(["javac", "-d", classes, "-cp", classpath, *files], capture_output=True, text=True, check=False)
        if compiled.returncode != 0:
            raise RuntimeError(f"javac failed:\n{compiled.stderr}")
        completed = subprocess.run(
            ["java", "-cp", os.pathsep.join([classes, classpath]), JAVA_MAIN],
            input="\n".join(parts) + "\n",
            capture_output=True,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"CcmoSelection.java failed:\n{completed.stderr}")
    return str(completed.stdout).splitlines()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jmetal-sources", required=True, type=Path, help="directory with the CCMO sources of jMetal PR #521")
    parser.add_argument("--classpath", required=True, help="jmetal-core with commons-lang3 and commons-math3")
    args = parser.parse_args()
    cases = _cases(np.random.default_rng(SEED))
    lines = iter(_jmetal(args.jmetal_sources, args.classpath, cases))
    for case in cases:
        case["fitness"] = [float(v) for v in next(lines).split()]
        case["helper_fitness"] = [float(v) for v in next(lines).split()]
        selections = []
        for size in case.pop("sizes"):
            for with_constraints in (True, False):
                selected = [int(i) for i in next(lines).split()]
                fitness = [float(v) for v in next(lines).split()]
                selections.append({"size": size, "with_constraints": with_constraints, "selected": selected, "fitness": fitness})
        case["selections"] = selections
    if next(lines, None) is not None:
        raise RuntimeError("CcmoSelection.java printed more lines than the cases need")
    payload = {
        "source": f"jMetal pull request #521, commit {JMETAL_COMMIT}, CCMOFitness and CCMOEnvironmentalSelection",
        "generator": "tools/ccmo_reference_selection.py",
        "cases": cases,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(cases)} cases written to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
