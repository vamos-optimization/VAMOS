"""Regenerate ``tests/engine/data/cmoeams_jmetal_reference.json`` with jMetal's CMOEA-MS.

The fixture holds, from jMetal's own classes on fixed inputs:

- ``populations``: for fixed objective vectors and constraint values (in jMetal's convention, a
  negative value is violated), the constraint violation (``CMOEAMSFitness.constraintViolation``),
  the shift-based density (``shiftBasedDensity``), the fitness of stage A and of stage B
  (``stageAFitness`` and ``stageBFitness``), the cosine distances between the objective vectors
  (``cosineDistances``), the rows that ``CMOEAMSSelection.environmentalSelection`` keeps with the
  fitness of each stage for several sizes, the rows that ``CMOEAMSSelection.truncation`` removes for
  several numbers, and the winners of ``CMOEAMSSelection.tournamentSelection`` with the fitness of
  each stage on fixed draws;
- ``points``: the fitness (``CMOEAMSFitness.fitness``) and the cosine distances of fixed points
  with a violation each, with zero vectors, ``NaN`` values and a single point;
- ``stage_rule``: the decisions of ``CMOEAMS.isStageB`` for fixed violations, ``lambda`` and
  evaluations;
- ``differential_evolution``: the children of jMetal's ``DifferentialEvolutionCrossover``
  ``RAND_1_BIN`` and ``RAND_1_EXP`` (with its clamping) for fixed vectors, bounds, CR, F and draws:
  for each child, the integers (``j_rand``, then the first variable of the exponential crossover)
  and the uniform draws, which a generator replays in order;
- ``sbx``: the children of jMetal's ``SBXCrossover`` for fixed pairs, bounds and draws: a draw per
  pair, and per variable of a crossed pair a draw that decides whether it is crossed and, when it is
  and its values differ, a draw for the spread and one for the swap of the two values, which the
  fixture stores as VAMOS draws them, in grids.

``tests/engine/test_cmoeams.py`` checks VAMOS's CMOEA-MS against them.

Run from a checkout::

    python tools/cmoeams_reference_values.py --jmetal-sources <dir> --classpath "<jars>"

``--jmetal-sources`` is a directory with the classes of the package
``org.uma.jmetal.algorithm.multiobjective.cmoeams`` of jMetal pull request #526 at commit
``JMETAL_COMMIT`` (``JMETAL_SOURCES``). ``--classpath`` lists, separated by the platform's path
separator, jMetal's ``jmetal-core`` 7.10 with commons-lang3 and commons-math3. The classes are
package-private, so the tool compiles them with ``tools/jmetal_values/CmoeamsReference.java``,
which shares their package, and runs it with Java 17 or later (``javac`` and ``java`` on the path).
Non-finite values are stored as the strings ``"inf"``, ``"-inf"`` and ``"nan"``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "engine" / "data" / "cmoeams_jmetal_reference.json"
JAVA_HELPER = ROOT / "tools" / "jmetal_values" / "CmoeamsReference.java"
JAVA_MAIN = "org.uma.jmetal.algorithm.multiobjective.cmoeams.CmoeamsReference"
JMETAL_SOURCES = ("CMOEAMS.java", "CMOEAMSBuilder.java", "CMOEAMSFitness.java", "CMOEAMSSelection.java")
JMETAL_COMMIT = "4d5f574f3183e6a015d65b1a1fad306db93ff95c"
SEED = 20261115
VARIANTS = ("rand_1_bin", "rand_1_exp")
MAX_STORED_DISTANCES = 30


def _java(value: float) -> str:
    """A float as Java's ``Double.parseDouble`` reads it, exactly."""
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return repr(float(value))


def _encoded(values: Any) -> Any:
    """Floats for JSON, with ``"inf"``, ``"-inf"`` and ``"nan"`` for the non-finite ones."""
    if isinstance(values, (list, tuple, np.ndarray)):
        return [_encoded(v) for v in values]
    value = float(values)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _constraints(rng: np.random.Generator, n: int, k: int, feasible: float, *, integer: bool = False) -> np.ndarray:
    """Constraint values in jMetal's convention: the rows drawn as feasible are positive, the others mixed."""
    C = rng.integers(-3, 3, size=(n, k)).astype(float) if integer else (rng.random((n, k)) - 0.6).round(6)
    C[rng.random(n) < feasible] = 1.0
    return C


def _population_case(rng: np.random.Generator, label: str, F: np.ndarray, C: np.ndarray) -> dict[str, Any]:
    n = F.shape[0]
    sizes = sorted({0, 1, n // 3, n // 2, (2 * n) // 3, n - 1, n})
    removed = sorted({0, 1, n // 4, n // 2, n - 1, n})
    draws = rng.integers(0, n, size=(120, 2))
    draws[:10, 1] = draws[:10, 0]
    return {
        "label": label,
        "F": _encoded(F.tolist()),
        "constraints": _encoded(C.tolist()),
        "sizes": sizes,
        "removed": removed,
        "draws": draws.tolist(),
    }


def _population_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def add(label: str, F: np.ndarray, C: np.ndarray) -> None:
        cases.append(_population_case(rng, label, F, C))

    # The example of jMetal's CMOEAMSFitnessTest, with the fourth row violating its constraint by 0.5.
    add("handmade", np.array([[0.0, 4.0], [2.0, 0.0], [1.0, 2.0], [2.0, 4.0]]), np.array([[1.0], [1.0], [1.0], [-0.5]]))
    add("random-2obj-2con", rng.random((20, 2)).round(6), _constraints(rng, 20, 2, 0.5))
    add("random-3obj-3con", rng.random((30, 3)).round(6), _constraints(rng, 30, 3, 0.5))
    add("grid-ties", np.round(rng.random((24, 2)) * 4) / 4, _constraints(rng, 24, 2, 0.4, integer=True))
    add("grid-3obj-ties", np.round(rng.random((30, 3)) * 2) / 2, _constraints(rng, 30, 1, 0.6, integer=True))
    add("all-feasible", rng.random((24, 2)).round(6), np.ones((24, 1)))
    add("all-infeasible", rng.random((16, 2)).round(6), -1.0 - rng.random((16, 2)).round(6))
    add("without-constraints", rng.random((18, 2)).round(6), np.zeros((18, 0)))
    # A constraint that no row violates: 0 / 0 in PlatEMO, which contributes zero.
    C = _constraints(rng, 15, 3, 0.4)
    C[:, 1] = np.abs(C[:, 1]) + 0.1
    add("unviolated-constraint", rng.random((15, 2)).round(6), C)
    # A constant objective: NaN densities, so NaN points [SDE, CV] in stage A.
    F = rng.random((12, 2)).round(6)
    F[:, 0] = 0.5
    add("constant-objective", F, _constraints(rng, 12, 1, 0.5))
    F = rng.random((12, 3)).round(6)
    F[:, 2] = 2.0
    add("constant-objective-all-feasible", F, np.ones((12, 1)))
    # Zero objective vectors: NaN cosine distances in stage B and in the truncation.
    F = rng.random((14, 2)).round(6)
    F[[2, 7, 11]] = 0.0
    add("zero-vectors", F, np.ones((14, 1)))
    F = rng.random((14, 3)).round(6)
    F[[0, 5]] = 0.0
    add("zero-vectors-3obj", F, _constraints(rng, 14, 2, 0.6))
    # NaN values: a NaN objective, a NaN constraint (no violation, as max(NaN, 0) of MATLAB).
    F = rng.random((10, 2)).round(6)
    F[3, 1] = math.nan
    C = _constraints(rng, 10, 2, 0.4)
    C[4, 0] = math.nan
    C[0, 1] = math.nan
    add("nan-values", F, C)
    # An infinite violation: inf / inf is NaN, which zeroes the constraint when it is in the first row.
    C = _constraints(rng, 8, 2, 0.3)
    C[0, 0] = -math.inf
    C[5, 1] = -math.inf
    add("infinite-violation", rng.random((8, 2)).round(6), C)
    F = rng.random((10, 2)).round(6)
    C = _constraints(rng, 10, 1, 0.5)
    add("duplicates", np.vstack([F, F[2:7]]), np.vstack([C, C[2:7]]))
    add("negative-objectives", (rng.random((16, 2)) * 2.0 - 1.0).round(6), _constraints(rng, 16, 1, 0.5))
    add("two", np.array([[0.3, 0.7], [0.1, 0.9]]), np.array([[-0.5], [1.0]]))
    add("three", np.array([[0.0, 1.0], [1.0, 0.0], [0.25, 0.25]]), np.array([[1.0], [1.0], [1.0]]))
    add("5obj", rng.random((25, 5)).round(6), _constraints(rng, 25, 2, 0.6))
    angles = rng.random(60) * np.pi / 2
    front = np.stack([np.cos(angles), np.sin(angles)], axis=1).round(6)
    add("dense-front", front, np.ones((60, 1)))
    # Unions of 200 as in a run: a noisy front, partly feasible.
    x = rng.random(200)
    F = np.stack([x, 1.0 - np.sqrt(x)], axis=1) + rng.random((200, 2)) * 0.05
    add("union-200", F.round(6), _constraints(rng, 200, 2, 0.6))
    F = rng.random((200, 3))
    F = (F / F.sum(axis=1, keepdims=True) + rng.random((200, 3)) * 0.02).round(6)
    add("union-200-3obj", F, _constraints(rng, 200, 3, 0.8))
    return cases


def _points_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def add(label: str, points: np.ndarray, violation: np.ndarray) -> None:
        cases.append({"label": label, "points": _encoded(points.tolist()), "violation": _encoded(violation.tolist())})

    add("single", np.array([[1.0, 2.0]]), np.array([0.0]))
    add("lower-violation-dominates", np.array([[5.0, 5.0], [1.0, 1.0]]), np.array([0.1, 0.2]))
    add("non-dominated", np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.6], [2.0, 2.0]]), np.zeros(4))
    add("zero-vectors", np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [2.0, 0.5], [0.5, 2.0]]), np.zeros(5))
    add(
        "nan-components",
        np.array([[math.nan, 0.0], [0.3, 0.0], [math.nan, 0.5], [0.4, 0.2], [0.2, 0.1]]),
        np.array([0.0, 0.0, 0.5, 0.2, 0.0]),
    )
    add("nan-violation", rng.random((8, 2)).round(6), np.array([0.0, math.nan, 0.2, 0.2, 0.0, math.nan, 0.1, 0.0]))
    add(
        "sde-cv-points",
        np.column_stack([1.0 / (2.0 + rng.random(20).round(3)), np.where(rng.random(20) < 0.5, 0.0, rng.random(20).round(3))]),
        np.zeros(20),
    )
    add("ties", np.round(rng.random((16, 2)) * 3) / 3, np.round(rng.random(16) * 2) / 4)
    add("negative", (rng.random((12, 3)) * 2.0 - 1.0).round(6), rng.random(12).round(1))
    return cases


def _stage_rule_cases() -> list[list[Any]]:
    violations = (
        [0.0, 0.0, 0.0, 0.5],
        [0.0, 0.0, 0.1, 0.2],
        [0.0, 0.0, math.nan, 0.5],
        [0.0] * 5,
        [0.2, 0.4],
        [0.0, 0.3, 0.0],
    )
    cases: list[list[Any]] = []
    for violation in violations:
        for threshold in (0.0, 0.5, 0.6, 0.75, 1.0, 0.49999999999999994):
            for evaluations, budget in (
                (999, 10000),
                (1000, 10000),
                (2499, 25000),
                (2500, 25000),
                (20000, 200000),
                (15, 150),
                (16, 150),
                (0, 3),
                (1, 3),
            ):
                cases.append([violation, threshold, evaluations, budget])
    return cases


def _de_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    lower = np.array([0.0, -5.0, 0.0, 1.0, -1.0, 0.0])
    upper = np.array([1.0, 5.0, 10.0, 2.0, 1.0, 1.0])
    n = lower.size
    for variant, cr, f in ((0, 1.0, 0.5), (0, 0.5, 0.8), (0, 0.0, 0.5), (1, 1.0, 0.5), (1, 0.5, 0.5), (1, 0.9, 1.0), (1, 0.0, 0.5)):
        rows = 12
        vectors = lower + rng.random((rows, 3, n)) * (upper - lower)
        draws = rng.random((rows, n)).round(3)
        draws[0, :] = cr  # A draw equal to CR does not cross.
        positions = rng.integers(0, n, size=(rows, 2))
        cases.append(
            {
                "label": f"{VARIANTS[variant]}-cr{cr}-f{f}",
                "variant": VARIANTS[variant],
                "cr": cr,
                "f": f,
                "lower": lower.tolist(),
                "upper": upper.tolist(),
                "vectors": vectors.tolist(),
                "positions": positions.tolist(),
                "draws": draws.tolist(),
            }
        )
    return cases


def _sbx_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    lower = np.array([0.0, -5.0, 0.0, 1.0, -1.0, 0.0])
    upper = np.array([1.0, 5.0, 10.0, 2.0, 1.0, 1.0])
    for prob, eta in ((1.0, 20.0), (0.7, 20.0), (1.0, 5.0)):
        pairs = 10
        parents = lower + rng.random((pairs, 2, lower.size)) * (upper - lower)
        parents[0, 1, :3] = parents[0, 0, :3]  # Equal values are not crossed.
        pair_draws = rng.random(pairs)
        pair_draws[1] = prob  # A draw equal to the probability crosses.
        var_draws = rng.random((pairs, lower.size))
        var_draws[2, :2] = 0.5  # A draw of 0.5 crosses the variable.
        spread_draws = rng.random((pairs, lower.size))
        swap_draws = rng.random((pairs, lower.size))
        swap_draws[3, :2] = 0.5  # A draw of 0.5 swaps the values.
        sequence: list[float] = []
        for p in range(pairs):
            sequence.append(float(pair_draws[p]))
            if pair_draws[p] > prob:
                continue
            for j in range(lower.size):
                sequence.append(float(var_draws[p, j]))
                if var_draws[p, j] <= 0.5 and abs(parents[p, 0, j] - parents[p, 1, j]) > 1.0e-14:
                    sequence.extend([float(spread_draws[p, j]), float(swap_draws[p, j])])
        cases.append(
            {
                "label": f"prob{prob:g}-eta{eta:g}",
                "prob": prob,
                "eta": eta,
                "lower": lower.tolist(),
                "upper": upper.tolist(),
                "parents": parents.tolist(),
                "pair_draws": pair_draws.tolist(),
                "var_draws": var_draws.tolist(),
                "spread_draws": spread_draws.tolist(),
                "swap_draws": swap_draws.tolist(),
                "sequence": sequence,
            }
        )
    return cases


def _rows(values: Any) -> list[str]:
    return [" ".join(_java(float(v)) for v in row) for row in values]


def _jmetal(sources: Path, classpath: str, payload: dict[str, Any]) -> list[str]:
    parts: list[str] = [str(len(payload["populations"]))]
    for case in payload["populations"]:
        n, m, k = len(case["F"]), len(case["F"][0]), len(case["constraints"][0])
        parts.append(f"{n} {m} {k}")
        parts.extend(_rows(case["F"]))
        parts.extend(_rows(case["constraints"]))
        parts.append(" ".join(map(str, [len(case["sizes"]), *case["sizes"]])))
        parts.append(" ".join(map(str, [len(case["removed"]), *case["removed"]])))
        parts.append(str(len(case["draws"])))
        parts.extend(f"{a} {b}" for a, b in case["draws"])
    parts.append(str(len(payload["points"])))
    for case in payload["points"]:
        parts.append(f"{len(case['points'])} {len(case['points'][0])}")
        parts.extend(_rows(case["points"]))
        parts.append(" ".join(_java(float(v)) for v in case["violation"]))
    parts.append(str(len(payload["stage_rule"])))
    for violation, threshold, evaluations, budget in payload["stage_rule"]:
        parts.append(" ".join([str(len(violation)), *(_java(v) for v in violation), _java(threshold), str(evaluations), str(budget)]))
    parts.append(str(len(payload["differential_evolution"])))
    for case in payload["differential_evolution"]:
        rows, n = len(case["vectors"]), len(case["lower"])
        parts.append(f"{VARIANTS.index(case['variant'])} {rows} {n} {_java(case['cr'])} {_java(case['f'])}")
        parts.append(" ".join(_java(v) for v in case["lower"]))
        parts.append(" ".join(_java(v) for v in case["upper"]))
        for group, positions, draws in zip(case["vectors"], case["positions"], case["draws"], strict=True):
            parts.extend(_rows(group))
            parts.append(" ".join(map(str, positions)))
            parts.append(" ".join(_java(v) for v in draws))
    parts.append(str(len(payload["sbx"])))
    for case in payload["sbx"]:
        parts.append(f"{len(case['parents'])} {len(case['lower'])} {_java(case['prob'])} {_java(case['eta'])}")
        parts.append(" ".join(_java(v) for v in case["lower"]))
        parts.append(" ".join(_java(v) for v in case["upper"]))
        for pair in case["parents"]:
            parts.extend(_rows(pair))
        parts.append(str(len(case["sequence"])))
        parts.append(" ".join(_java(v) for v in case["sequence"]))
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
        raise RuntimeError(f"CmoeamsReference.java failed:\n{completed.stderr}")
    return str(completed.stdout).split("\n")


def _indices(line: str) -> list[int]:
    return [int(v) for v in line.split()]


def _floats(line: str) -> list[Any]:
    return _encoded([float(v) for v in line.split()])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jmetal-sources", required=True, type=Path, help="directory with the CMOEA-MS sources of jMetal PR #526")
    parser.add_argument("--classpath", required=True, help="jmetal-core 7.10 with commons-lang3 and commons-math3")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    payload: dict[str, Any] = {
        "source": f"jMetal pull request #526, commit {JMETAL_COMMIT}, package org.uma.jmetal.algorithm.multiobjective.cmoeams; jMetal 7.10",
        "generator": "tools/cmoeams_reference_values.py",
        "populations": _population_cases(rng),
        "points": _points_cases(rng),
        "stage_rule": _stage_rule_cases(),
        "differential_evolution": _de_cases(rng),
        "sbx": _sbx_cases(rng),
    }
    lines = iter(_jmetal(args.jmetal_sources, args.classpath, payload))
    for case in payload["populations"]:
        n = len(case["F"])
        case["violation"] = _floats(next(lines))
        case["density"] = _floats(next(lines))
        case["stage_a_fitness"] = _floats(next(lines))
        case["stage_b_fitness"] = _floats(next(lines))
        distances = [_floats(next(lines)) for _ in range(n)]
        # The matrices of the unions of 200 would make most of the file; their fitness and truncations check them.
        if n <= MAX_STORED_DISTANCES:
            case["cosine_distances"] = distances
        case["selected_a"], case["selected_b"] = [], []
        for _ in case["sizes"]:
            case["selected_a"].append(_indices(next(lines)))
            case["selected_b"].append(_indices(next(lines)))
        case["truncated"] = [[bool(v) for v in _indices(next(lines))] for _ in case["removed"]]
        case["winners_a"] = _indices(next(lines))
        case["winners_b"] = _indices(next(lines))
    for case in payload["points"]:
        case["fitness"] = _floats(next(lines))
        case["cosine_distances"] = [_floats(next(lines)) for _ in case["points"]]
    decisions = [bool(v) for v in _indices(next(lines))]
    payload["stage_rule"] = [
        {"violation": _encoded(v), "threshold": t, "evaluations": e, "max_evaluations": b, "stage_b": decision}
        for (v, t, e, b), decision in zip(payload["stage_rule"], decisions, strict=True)
    ]
    for case in payload["differential_evolution"]:
        case["children"] = [_floats(next(lines)) for _ in case["vectors"]]
        case["draws_used"] = _indices(next(lines))
    for case in payload["sbx"]:
        case["children"] = [[_floats(next(lines)), _floats(next(lines))] for _ in case["parents"]]
        case["draws_used"] = int(next(lines))
    if any(line.strip() for line in lines):
        raise RuntimeError("CmoeamsReference.java printed more lines than the cases need")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    counts = ", ".join(
        f"{len(payload[key])} {key.replace('_', ' ')}" for key in ("populations", "points", "stage_rule", "differential_evolution", "sbx")
    )
    print(f"{counts} cases written to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
