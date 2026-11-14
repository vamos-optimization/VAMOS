"""Regenerate ``tests/engine/data/top_jmetal_reference.json`` with jMetal's ToP.

The fixture holds, from jMetal's own classes on fixed inputs:

- ``selection``: for fixed populations (objective vectors and a violation per row), the front
  numbers (``ToPSelection.frontNumbers``), the crowding distances
  (``ToPSelection.crowdingDistances``), the rows that the truncation keeps for several sizes
  (``ToPSelection.select``, in jMetal's order) and the winners of binary tournaments on fixed
  pairs of rows (``ToPSelection.binaryTournament``);
- ``differential_evolution``: the trial vectors of ``ToPDifferentialEvolution`` (its three
  operators, after the clamping and with a mutation that changes nothing) for fixed vectors,
  bounds, CR, F and crossover draws, which a generator replays in order (one per variable);
- ``polynomial_mutation``: the vectors mutated by jMetal's ``PolynomialMutation``, the default
  mutation of ToP, for fixed vectors, bounds and draws (for each variable a draw that decides
  whether it mutates and, when it does and its bounds differ, a draw for the perturbation);
- ``phase_one``: for fixed populations and offspring with several constraints (in jMetal's
  convention, a negative value is violated), the violations (``ToP.constraintViolation``), the
  mean objectives, the best solution of phase 1, ``Pf``, ``delta`` before and after the
  one-to-one replacement, and which offspring replace their parents;
- ``phase_rule``: the decisions of ``ToP.phaseOneContinues`` for fixed delta, Pf and evaluations.

``tests/engine/test_top.py`` checks VAMOS's ToP against them.

Run from a checkout::

    python tools/top_reference_values.py --jmetal-sources <dir> --classpath "<jars>"

``--jmetal-sources`` is a directory with the classes of the package
``org.uma.jmetal.algorithm.multiobjective.top`` of jMetal pull request #525 at commit
``JMETAL_COMMIT`` (``JMETAL_SOURCES``). ``--classpath`` lists, separated by the platform's path
separator, jMetal's ``jmetal-core`` (7.10 or later) with commons-lang3 and commons-math3. The
classes are package-private, so the tool compiles them with ``tools/jmetal_values/TopReference.java``,
which shares their package, and runs it with Java 17 or later (``javac`` and ``java`` on the
path). Violations are stored as non-negative numbers, the VAMOS ``compute_violation`` of a single
constraint ``g = violation``; the constraint values of ``phase_one`` are jMetal's, so VAMOS's are
their negation. Non-finite values are stored as the strings ``"inf"``, ``"-inf"`` and ``"nan"``.
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
OUTPUT = ROOT / "tests" / "engine" / "data" / "top_jmetal_reference.json"
JAVA_HELPER = ROOT / "tools" / "jmetal_values" / "TopReference.java"
JAVA_MAIN = "org.uma.jmetal.algorithm.multiobjective.top.TopReference"
JMETAL_SOURCES = ("ToP.java", "ToPBuilder.java", "ToPDifferentialEvolution.java", "ToPSelection.java")
JMETAL_COMMIT = "e7fdd9974fdf7cff01f925960e1891d7ede6ba8d"
SEED = 20261114
OPERATORS = ("current_to_rand_1", "rand_to_best_1", "rand_1")


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


def _population(
    rng: np.random.Generator, n: int, m: int, *, feasible: float, grid: float | None = None, integer: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    F = rng.random((n, m))
    F = np.round(F / grid) * grid if grid else F.round(6)
    violation = rng.integers(1, 4, size=n).astype(float) if integer else rng.random(n).round(6) + 1e-3
    violation[rng.random(n) < feasible] = 0.0
    return F, violation


def _selection_case(
    rng: np.random.Generator, label: str, F: np.ndarray, violation: np.ndarray, sizes: list[int] | None = None
) -> dict[str, Any]:
    n = F.shape[0]
    if sizes is None:
        sizes = sorted({0, 1, n // 3, n // 2, (2 * n) // 3, n - 1, n})
    draws = rng.integers(0, n, size=(120, 2))
    draws[:10, 1] = draws[:10, 0]
    return {"label": label, "F": F.tolist(), "violation": violation.tolist(), "sizes": sizes, "draws": draws.tolist()}


def _selection_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    # a dominates b, c and a are non-dominated, d duplicates b; e and f are infeasible with the same violation, g has a larger one.
    F = np.array([[1.0, 3.0], [2.0, 4.0], [3.0, 1.0], [2.0, 4.0], [0.5, 0.5], [0.0, 5.0], [0.1, 0.1], [1.5, 2.0]])
    violation = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0])
    case = _selection_case(rng, "handmade", F, violation, sizes=list(range(9)))
    case["draws"] = [[a, b] for a in range(8) for b in range(8)]
    cases.append(case)
    cases.append(_selection_case(rng, "random-2obj-partly-feasible", *_population(rng, 20, 2, feasible=0.5)))
    cases.append(_selection_case(rng, "random-3obj", *_population(rng, 30, 3, feasible=0.6)))
    cases.append(_selection_case(rng, "grid-all-feasible", *_population(rng, 24, 2, feasible=1.0, grid=0.25)))
    cases.append(_selection_case(rng, "grid-3obj-ties", *_population(rng, 30, 3, feasible=0.7, grid=0.5)))
    cases.append(_selection_case(rng, "integer-violations-all-infeasible", *_population(rng, 15, 2, feasible=0.0, integer=True)))
    cases.append(
        _selection_case(rng, "integer-violations-partly-feasible", *_population(rng, 20, 2, feasible=0.3, integer=True, grid=0.25))
    )
    # Fronts with a flat objective: NaN distances, which an infinite one in a later objective can overwrite.
    shares = rng.random(8).round(6)
    flat = np.stack([np.full(8, 0.25), shares, 1.0 - shares], axis=1)
    flat2 = np.stack([shares, np.full(8, 0.5)], axis=1)
    cases.append(_selection_case(rng, "flat-objective-3obj", flat, np.zeros(8)))
    cases.append(_selection_case(rng, "flat-objective-2obj", flat2, np.zeros(8)))
    mixed = np.vstack([flat2, rng.random((10, 2)).round(6)])
    mixed_violation = np.concatenate([np.zeros(8), rng.integers(1, 3, size=10).astype(float)])
    cases.append(_selection_case(rng, "flat-front-and-infeasible", mixed, mixed_violation))
    same = np.ones((6, 2)) * 0.5
    cases.append(_selection_case(rng, "identical-rows", same, np.zeros(6)))
    F, violation = _population(rng, 10, 2, feasible=0.6)
    duplicated = np.vstack([F, F[2:7]])
    cases.append(_selection_case(rng, "duplicates", duplicated, np.concatenate([violation, violation[2:7]])))
    cases.append(_selection_case(rng, "single", np.array([[0.3, 0.7]]), np.array([0.0])))
    cases.append(_selection_case(rng, "two-infeasible", np.array([[0.3, 0.7], [0.1, 0.9]]), np.array([0.5, 0.5])))
    cases.append(_selection_case(rng, "5obj", *_population(rng, 25, 5, feasible=0.6)))
    angles = rng.random(60) * np.pi / 2
    front = np.stack([np.cos(angles), np.sin(angles)], axis=1).round(6)
    cases.append(_selection_case(rng, "dense-front", front, np.zeros(60), sizes=[0, 10, 30, 59, 60]))
    F, violation = _population(rng, 200, 2, feasible=0.5)
    cases.append(_selection_case(rng, "union-200", F, violation, sizes=[0, 50, 100, 150, 200]))
    F, violation = _population(rng, 200, 3, feasible=0.9, grid=0.1)
    cases.append(_selection_case(rng, "union-200-3obj-grid", F, violation, sizes=[0, 50, 100, 150, 200]))
    return cases


def _de_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    lower = np.array([0.0, -5.0, 0.0, 1.0, -1.0, 0.0])
    upper = np.array([1.0, 5.0, 10.0, 2.0, 1.0, 1.0])
    for operator, cr, f in (
        (0, 1.0, 0.5),
        (0, 0.5, 0.8),
        (1, 1.0, 0.5),
        (1, 0.3, 0.9),
        (2, 1.0, 0.5),
        (2, 0.6, 0.5),
        (2, 0.0, 0.5),
        (0, 0.9, 1.0),
    ):
        trials = 12
        vectors = 3 if operator == 2 else 4
        groups = lower + rng.random((trials, vectors, lower.size)) * (upper - lower)
        draws = rng.random((trials, lower.size)).round(3)
        draws[0, :] = cr  # A draw equal to CR does not cross.
        cases.append(
            {
                "label": f"{OPERATORS[operator]}-cr{cr}-f{f}",
                "operator": OPERATORS[operator],
                "cr": cr,
                "f": f,
                "lower": lower.tolist(),
                "upper": upper.tolist(),
                "vectors": groups.tolist(),
                "draws": draws.tolist(),
            }
        )
    return cases


def _pm_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    lower = np.array([0.0, -5.0, 0.0, 1.0, 2.0, 0.0, -1.0])
    upper = np.array([1.0, 5.0, 10.0, 2.0, 2.0, 1.0, 1.0])
    for prob, eta in ((1.0 / 7.0, 20.0), (1.0, 5.0), (0.5, 100.0), (1.0, 20.0)):
        rows = 12
        X = lower + rng.random((rows, lower.size)) * (upper - lower)
        X[0] = lower
        X[1] = upper
        mask = rng.random((rows, lower.size))
        mask[2, :3] = prob  # A draw equal to the probability mutates.
        delta = rng.random((rows, lower.size))
        delta[3, :] = 0.5
        delta[4, :] = 0.0
        sequence: list[float] = []
        for r in range(rows):
            for j in range(lower.size):
                sequence.append(float(mask[r, j]))
                if mask[r, j] <= prob and lower[j] != upper[j]:
                    sequence.append(float(delta[r, j]))
        cases.append(
            {
                "label": f"prob{prob:.3g}-eta{eta:g}",
                "prob": prob,
                "eta": eta,
                "lower": lower.tolist(),
                "upper": upper.tolist(),
                "X": X.tolist(),
                "mask_draws": mask.tolist(),
                "delta_draws": delta.tolist(),
                "sequence": sequence,
            }
        )
    return cases


def _phase_one_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    def add(label: str, F: np.ndarray, C: np.ndarray, F_off: np.ndarray, C_off: np.ndarray, delta: float = 1.0) -> None:
        cases.append(
            {
                "label": label,
                "delta": delta,
                "F": F.tolist(),
                "constraints": C.tolist(),
                "offspring_F": F_off.tolist(),
                "offspring_constraints": C_off.tolist(),
            }
        )

    def constraints(n: int, k: int, feasible: float, integer: bool = False) -> np.ndarray:
        C = rng.integers(-3, 3, size=(n, k)).astype(float) if integer else (rng.random((n, k)) - 0.6).round(6)
        C[rng.random(n) < feasible] = 1.0
        return C

    for label, n, m, k, feasible in (
        ("random-2obj-1con", 10, 2, 1, 0.5),
        ("random-3obj-3con", 12, 3, 3, 0.5),
        ("many-constraints", 10, 2, 9, 0.3),
        ("population-100", 100, 2, 2, 0.4),
    ):
        add(label, rng.random((n, m)).round(6), constraints(n, k, feasible), rng.random((n, m)).round(6), constraints(n, k, feasible))
    # Fewer than three feasible solutions: delta keeps its value.
    add("two-feasible", rng.random((8, 2)).round(6), constraints(8, 1, 0.0), rng.random((8, 2)).round(6), constraints(8, 1, 0.0), delta=0.7)
    C = constraints(8, 1, 0.0)
    C[:2] = 1.0
    add("keeps-delta", rng.random((8, 2)).round(6), C, rng.random((8, 2)).round(6), constraints(8, 1, 0.2), delta=0.45)
    # A flat objective among the feasible solutions: delta is NaN.
    F = rng.random((9, 2)).round(6)
    F[:, 0] = 0.4
    add("flat-objective", F, np.ones((9, 1)), F[::-1].copy(), np.ones((9, 1)))
    # Ties: equal violations and means, integer data.
    F = np.round(rng.random((12, 2)) * 4) / 4
    add("ties", F, constraints(12, 2, 0.3, integer=True), np.round(rng.random((12, 2)) * 4) / 4, constraints(12, 2, 0.3, integer=True))
    # All feasible on a linear front: delta is (almost) zero.
    x = rng.random(12).round(6)
    line = np.stack([x, 1.0 - x], axis=1)
    add("linear-front", line, np.ones((12, 1)), line[::-1].copy(), np.ones((12, 1)))
    # Equal means and violations with the offspring: no replacement.
    F = rng.random((6, 3)).round(6)
    C = constraints(6, 2, 0.5)
    add("identical-offspring", F, C, F.copy(), C.copy())
    # The lowest mean objective is shared: the first one is the best.
    F = np.array([[2.0, 2.0], [1.0, 1.0], [0.5, 1.5], [3.0, 0.0], [1.5, 0.5]])
    add("shared-best", F, np.array([[1.0], [-5.0], [1.0], [1.0], [1.0]]), F[::-1].copy(), np.ones((5, 1)))
    return cases


def _phase_rule_cases() -> list[list[Any]]:
    third = 1.0 / 3.0
    cases: list[list[Any]] = []
    for delta in (0.2, 0.19999999999999998, 1.0, 0.0, math.nan, math.inf):
        for ratio in (third, 0.34, 0.0, 1.0, 0.3333333333333333):
            for evaluations, budget in (
                (100, 1000),
                (900, 1000),
                (901, 1000),
                (22500, 25000),
                (22501, 25000),
                (180, 200),
                (190, 200),
                (89, 99),
                (90, 99),
            ):
                cases.append([delta, ratio, evaluations, budget])
    return cases


def _jmetal(sources: Path, classpath: str, payload: dict[str, Any]) -> list[str]:
    parts: list[str] = [str(len(payload["selection"]))]
    for case in payload["selection"]:
        F = np.asarray(case["F"])
        parts.append(f"{F.shape[0]} {F.shape[1]}")
        parts.extend(" ".join(_java(v) for v in (*row, cv)) for row, cv in zip(case["F"], case["violation"], strict=True))
        parts.append(" ".join([str(len(case["sizes"])), *map(str, case["sizes"])]))
        parts.append(str(len(case["draws"])))
        parts.extend(f"{a} {b}" for a, b in case["draws"])
    parts.append(str(len(payload["differential_evolution"])))
    for case in payload["differential_evolution"]:
        vectors = np.asarray(case["vectors"])
        parts.append(f"{OPERATORS.index(case['operator'])} {vectors.shape[0]} {vectors.shape[2]} {_java(case['cr'])} {_java(case['f'])}")
        parts.append(" ".join(_java(v) for v in case["lower"]))
        parts.append(" ".join(_java(v) for v in case["upper"]))
        parts.extend(" ".join(_java(v) for v in vector) for group in case["vectors"] for vector in group)
        parts.extend(" ".join(_java(v) for v in row) for row in case["draws"])
    parts.append(str(len(payload["polynomial_mutation"])))
    for case in payload["polynomial_mutation"]:
        X = np.asarray(case["X"])
        parts.append(f"{X.shape[0]} {X.shape[1]} {_java(case['prob'])} {_java(case['eta'])}")
        parts.append(" ".join(_java(v) for v in case["lower"]))
        parts.append(" ".join(_java(v) for v in case["upper"]))
        parts.extend(" ".join(_java(v) for v in row) for row in case["X"])
        parts.append(str(len(case["sequence"])))
        parts.append(" ".join(_java(v) for v in case["sequence"]))
    parts.append(str(len(payload["phase_one"])))
    for case in payload["phase_one"]:
        F, C = np.asarray(case["F"]), np.asarray(case["constraints"])
        parts.append(f"{F.shape[0]} {F.shape[1]} {C.shape[1]} {_java(case['delta'])}")
        for key_F, key_C in (("F", "constraints"), ("offspring_F", "offspring_constraints")):
            parts.extend(" ".join(_java(v) for v in (*row, *con)) for row, con in zip(case[key_F], case[key_C], strict=True))
    parts.append(str(len(payload["phase_rule"])))
    parts.extend(" ".join([_java(d), _java(r), str(e), str(b)]) for d, r, e, b in payload["phase_rule"])
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
        raise RuntimeError(f"TopReference.java failed:\n{completed.stderr}")
    return str(completed.stdout).split("\n")


def _indices(line: str) -> list[int]:
    return [int(v) for v in line.split()]


def _floats(line: str) -> list[float]:
    return [float(v) for v in line.split()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jmetal-sources", required=True, type=Path, help="directory with the ToP sources of jMetal PR #525")
    parser.add_argument("--classpath", required=True, help="jmetal-core with commons-lang3 and commons-math3")
    args = parser.parse_args()
    rng = np.random.default_rng(SEED)
    payload: dict[str, Any] = {
        "source": f"jMetal pull request #525, commit {JMETAL_COMMIT}, package org.uma.jmetal.algorithm.multiobjective.top",
        "generator": "tools/top_reference_values.py",
        "selection": _selection_cases(rng),
        "differential_evolution": _de_cases(rng),
        "polynomial_mutation": _pm_cases(rng),
        "phase_one": _phase_one_cases(rng),
        "phase_rule": _phase_rule_cases(),
    }
    lines = iter(_jmetal(args.jmetal_sources, args.classpath, payload))
    for case in payload["selection"]:
        case["fronts"] = _indices(next(lines))
        case["crowding"] = _encoded(_floats(next(lines)))
        case["selected"] = [_indices(next(lines)) for _ in case["sizes"]]
        case["winners"] = _indices(next(lines))
    for case in payload["differential_evolution"]:
        case["trials"] = [_floats(next(lines)) for _ in case["vectors"]]
        case["draws_used"] = int(next(lines))
    for case in payload["polynomial_mutation"]:
        case["mutated"] = [_floats(next(lines)) for _ in case["X"]]
        case["draws_used"] = int(next(lines))
    for case in payload["phase_one"]:
        case["violation"] = _floats(next(lines))
        case["offspring_violation"] = _floats(next(lines))
        case["mean"] = _floats(next(lines))
        case["best"] = int(next(lines))
        case["feasible_ratio"] = float(next(lines))
        case["delta_before"] = _encoded(float(next(lines)))
        case["replaces"] = [bool(v) for v in _indices(next(lines))]
        case["kept_offspring"] = [bool(v) for v in _indices(next(lines))]
        case["next_feasible_ratio"] = float(next(lines))
        case["next_delta"] = _encoded(float(next(lines)))
    decisions = [bool(v) for v in _indices(next(lines))]
    payload["phase_rule"] = [
        {"delta": _encoded(d), "feasible_ratio": r, "evaluations": e, "max_evaluations": b, "phase_one": decision}
        for (d, r, e, b), decision in zip(payload["phase_rule"], decisions, strict=True)
    ]
    if any(line.strip() for line in lines):
        raise RuntimeError("TopReference.java printed more lines than the cases need")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    counts = ", ".join(
        f"{len(payload[key])} {key.replace('_', ' ')}"
        for key in ("selection", "differential_evolution", "polynomial_mutation", "phase_one", "phase_rule")
    )
    print(f"{counts} cases written to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
