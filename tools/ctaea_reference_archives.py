"""Regenerate ``tests/engine/data/ctaea_jmetal_archives.json`` with jMetal's C-TAEA.

The fixture holds, from jMetal's own classes:

- the weight vectors of ``UniformWeightVectors`` for several requested sizes and objective counts;
- the winners of ``CTAEATournamentSelection`` on fixed archives and fixed draws, which a
  generator replays in order (a tournament consumes its third draw only when its first two
  solutions are infeasible);
- for the weight vectors of ``UniformWeightVectors`` and fixed convergence archives (CA),
  diversity archives (DA) and offspring (objective vectors and a constraint violation per row):
  the weight vectors themselves, the subregions (``Subregions.associate``)
  and Tchebycheff values (``Subregions.tchebycheffInOwnSubregion``) of the candidates ``CA +
  offspring``, the solution that the truncation of the CA would remove from them
  (``worstOfTheMostCrowdedSubregion``), the new CA (``ConvergenceArchiveUpdate``, rows of the
  candidates in jMetal's order), the new DA (``DiversityArchiveUpdate`` with the new CA, rows of
  ``DA + offspring``) and their ``MatingProportions``.

``tests/engine/test_ctaea.py`` checks VAMOS's C-TAEA against them.

Run from a checkout::

    python tools/ctaea_reference_archives.py --jmetal-sources <dir> --classpath "<jars>"

``--jmetal-sources`` is a directory with the classes of the package
``org.uma.jmetal.algorithm.multiobjective.ctaea.util`` of jMetal pull request #524 at commit
``JMETAL_COMMIT`` (``JMETAL_SOURCES``). ``--classpath`` lists, separated by the platform's path
separator, jMetal's ``jmetal-core`` (7.10 or later) with commons-lang3 and commons-math3. The
truncation's worst solution is package-private, so the tool compiles the classes with
``tools/jmetal_values/CtaeaArchives.java``, which shares their package, and runs it with Java 17
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
OUTPUT = ROOT / "tests" / "engine" / "data" / "ctaea_jmetal_archives.json"
JAVA_HELPER = ROOT / "tools" / "jmetal_values" / "CtaeaArchives.java"
JAVA_MAIN = "org.uma.jmetal.algorithm.multiobjective.ctaea.util.CtaeaArchives"
JMETAL_SOURCES = (
    "CTAEATournamentSelection.java",
    "ConstraintViolation.java",
    "ConvergenceArchiveUpdate.java",
    "DiversityArchiveUpdate.java",
    "MatingProportions.java",
    "NonDominatedSorting.java",
    "Subregions.java",
    "UniformWeightVectors.java",
)
JMETAL_COMMIT = "a19e9fa6a34fe63515a82f77da370ae14056d572"
SEED = 20261113
WEIGHT_CASES = (
    (1, 2),
    (12, 2),
    (100, 2),
    (2, 3),
    (3, 3),
    (5, 3),
    (10, 3),
    (91, 3),
    (92, 3),
    (100, 3),
    (4, 4),
    (50, 4),
    (30, 5),
    (105, 5),
    (30, 6),
    (156, 8),
    (275, 10),
)


def _rows(F: np.ndarray, violation: np.ndarray) -> dict[str, Any]:
    return {"F": F.tolist(), "violation": violation.tolist()}


def _population(
    rng: np.random.Generator, n: int, n_obj: int, *, feasible: float, integer_violation: bool = False, grid: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    F = rng.random((n, n_obj))
    F = np.round(F / grid) * grid if grid else F.round(6)
    violation = rng.integers(1, 4, size=n).astype(float) if integer_violation else rng.random(n).round(6) + 1e-3
    violation[rng.random(n) < feasible] = 0.0
    return F, violation


def _tournament_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    # a dominates b, c and a are non-dominated, d duplicates b, e and f are infeasible.
    F = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 1.0], [2.0, 4.0], [0.5, 0.5], [0.0, 5.0]])
    violation = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0])
    every = [[a, b, (5 * a + 3 * b) % 6] for a in range(6) for b in range(6)]
    cases.append({"label": "every-pair", **_rows(F, violation), "draws": every})
    for label, n, n_obj, feasible in (("random-3obj", 12, 3, 0.5), ("all-feasible", 10, 2, 1.0), ("all-infeasible", 10, 2, 0.0)):
        F, violation = _population(rng, n, n_obj, feasible=feasible, grid=0.25)
        draws = rng.integers(0, n, size=(150, 3)).tolist()
        cases.append({"label": label, **_rows(F, violation), "draws": draws})
    return cases


def _archive_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    empty = (np.empty((0, 2)), np.empty(0))

    def add(label: str, weights: tuple[int, int], convergence: Any, diversity: Any, offspring: Any) -> None:
        cases.append(
            {
                "label": label,
                "requested": weights[0],
                "n_obj": weights[1],
                "convergence": _rows(*convergence),
                "diversity": _rows(*diversity),
                "offspring": _rows(*offspring),
            }
        )

    def population(n: int, n_obj: int, **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        return _population(rng, n, n_obj, **kwargs)

    w10 = (10, 2)
    # The first update of both archives, from the initial population.
    add("init-feasible", w10, empty, empty, population(10, 2, feasible=1.0))
    add("init-partly-feasible", w10, empty, empty, population(10, 2, feasible=0.4))
    w15 = (15, 3)
    empty3 = (np.empty((0, 3)), np.empty(0))
    add("init-infeasible-integer-violations", w15, empty3, empty3, population(15, 3, feasible=0.0, integer_violation=True))
    # Exactly N feasible candidates.
    CA = population(10, 2, feasible=1.0)
    add("exactly-n-feasible", w10, CA, population(10, 2, feasible=0.5), population(10, 2, feasible=0.0))
    # More than N feasible candidates: the truncation.
    w20 = (20, 2)
    # Offspring near a linear front: the first front has more than N rows.
    near_front = rng.random(20).round(6)
    offspring = (np.stack([near_front, 1.0 - near_front], axis=1) * 0.6 + rng.random((20, 2)).round(6) * 0.05, np.zeros(20))
    add("truncation-2obj", w20, population(20, 2, feasible=1.0), population(20, 2, feasible=0.5), offspring)
    w91 = (100, 3)
    add("truncation-3obj", w91, population(91, 3, feasible=1.0), population(91, 3, feasible=0.3), population(91, 3, feasible=0.7))
    add(
        "truncation-grid",
        w10,
        population(10, 2, feasible=1.0, grid=0.25),
        population(10, 2, feasible=0.5, grid=0.25),
        population(10, 2, feasible=1.0, grid=0.25),
    )
    F, violation = population(12, 2, feasible=1.0)
    duplicated = (np.concatenate([F[:6], F[1:5], F[2:4]]), np.zeros(12))
    add("truncation-duplicates", w10, duplicated, population(10, 2, feasible=0.5), (F[6:], violation[6:]))
    same = (np.ones((10, 2)) * 0.5, np.zeros(10))
    add("truncation-identical-rows", w10, same, same, same)
    # A flat objective among non-dominated rows: their normalized objectives are NaN.
    shares = rng.random(20).round(6)
    flat = np.stack([np.full(20, 0.25), shares, 1.0 - shares], axis=1)
    add("truncation-flat-objective", (10, 3), (flat[:10], np.zeros(10)), population(10, 3, feasible=0.5), (flat[10:], np.zeros(10)))
    angles = np.sort(rng.random(100)) * np.pi / 2
    front = np.stack([np.cos(angles), np.sin(angles)], axis=1).round(6)
    add("truncation-dense-front", (50, 2), (front[:50], np.zeros(50)), population(50, 2, feasible=0.5), (front[50:], np.zeros(50)))
    # Fewer than N feasible candidates: the infeasible ones complete the CA.
    add(
        "fill-integer-violations-grid",
        w20,
        population(20, 2, feasible=0.2, integer_violation=True, grid=0.25),
        population(20, 2, feasible=0.2, grid=0.25),
        population(20, 2, feasible=0.2, integer_violation=True, grid=0.25),
    )
    w28 = (30, 3)
    add("fill-all-infeasible-3obj", w28, population(28, 3, feasible=0.0), population(28, 3, feasible=0.0), population(28, 3, feasible=0.0))
    add("fill-partly-feasible-3obj", w28, population(28, 3, feasible=0.3), population(28, 3, feasible=0.5), population(28, 3, feasible=0.3))
    # The last generation of an exact budget: fewer offspring than weight vectors.
    add("partial-offspring", w91, population(91, 3, feasible=0.8), population(91, 3, feasible=0.4), population(7, 3, feasible=0.6))
    # Offspring at the origin, whose cosine similarities are not defined, in the CA and the DA.
    F, violation = population(10, 2, feasible=0.5)
    F[3] = 0.0
    violation[3] = 0.0
    add("origin", w10, population(10, 2, feasible=0.5), population(10, 2, feasible=0.5), (F, violation))
    add(
        "two-layers-5obj",
        (30, 5),
        population(30, 5, feasible=0.5),
        population(30, 5, feasible=0.5),
        population(30, 5, feasible=0.6),
    )
    w100 = (100, 2)
    add("random-2obj", w100, population(100, 2, feasible=0.5), population(100, 2, feasible=0.5), population(100, 2, feasible=0.5))
    w5 = (5, 2)
    add(
        "few-weights-many-ties",
        w5,
        population(5, 2, feasible=0.6, grid=0.5),
        population(5, 2, feasible=0.6, grid=0.5),
        population(5, 2, feasible=0.6, grid=0.5),
    )
    return cases


def _solutions(parts: list[str], block: dict[str, Any]) -> None:
    parts.append(str(len(block["F"])))
    parts.extend(" ".join(repr(float(v)) for v in (*row, cv)) for row, cv in zip(block["F"], block["violation"], strict=True))


def _jmetal(sources: Path, classpath: str, tournaments: list[dict[str, Any]], archives: list[dict[str, Any]]) -> list[str]:
    parts = [str(len(WEIGHT_CASES))]
    parts.extend(f"{requested} {n_obj}" for requested, n_obj in WEIGHT_CASES)
    parts.append(str(len(tournaments)))
    for case in tournaments:
        parts.append(str(len(case["F"][0])))
        _solutions(parts, case)
        parts.append(str(len(case["draws"])))
        parts.extend(" ".join(str(int(d)) for d in draw) for draw in case["draws"])
    parts.append(str(len(archives)))
    for case in archives:
        parts.append(f"{case['requested']} {case['n_obj']}")
        for key in ("convergence", "diversity", "offspring"):
            _solutions(parts, case[key])
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
        raise RuntimeError(f"CtaeaArchives.java failed:\n{completed.stderr}")
    return str(completed.stdout).splitlines()


def _indices(line: str) -> list[int]:
    return [int(v) for v in line.split()]


def _floats(line: str) -> list[float]:
    return [float(v) for v in line.split()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jmetal-sources", required=True, type=Path, help="directory with the ctaea.util sources of jMetal PR #524")
    parser.add_argument("--classpath", required=True, help="jmetal-core with commons-lang3 and commons-math3")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    tournaments = _tournament_cases(rng)
    archives = _archive_cases(rng)
    lines = iter(_jmetal(args.jmetal_sources, args.classpath, tournaments, archives))
    weights = []
    for requested, n_obj in WEIGHT_CASES:
        count = int(next(lines))
        weights.append({"requested": requested, "n_obj": n_obj, "weights": [_floats(next(lines)) for _ in range(count)]})
    for case in tournaments:
        case["winners"] = _indices(next(lines))
    for case in archives:
        case["weights"] = [_floats(next(lines)) for _ in range(int(next(lines)))]
        case["regions"] = _indices(next(lines))
        case["tchebycheff"] = _floats(next(lines))
        case["worst"] = int(next(lines))
        case["convergence_update"] = _indices(next(lines))
        case["diversity_update"] = _indices(next(lines))
        case["proportions"] = _floats(next(lines))
    if next(lines, None) is not None:
        raise RuntimeError("CtaeaArchives.java printed more lines than the cases need")
    payload = {
        "source": f"jMetal pull request #524, commit {JMETAL_COMMIT}, package org.uma.jmetal.algorithm.multiobjective.ctaea.util",
        "generator": "tools/ctaea_reference_archives.py",
        "weight_vectors": weights,
        "tournaments": tournaments,
        "archives": archives,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(weights)} weight, {len(tournaments)} tournament and {len(archives)} archive cases written to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
