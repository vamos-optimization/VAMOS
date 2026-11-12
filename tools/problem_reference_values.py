"""Regenerate ``tests/foundation/data/problem_reference_values.json`` from independent implementations.

For each benchmark problem that another implementation also defines, the fixture holds three points
inside the VAMOS bounds and the objective and constraint values of that implementation there.
``tests/foundation/test_problem_reference_values.py`` checks VAMOS against them. The sources are
pymoo (ZDT, DTLZ, WFG, MW and the constrained DTLZ variants) and jMetal (UF, LZ09, LSMOP5-8, ZCAT,
RE and RWA, and the constrained suites below from jMetal's ``main``). Problems whose counterpart is not
the same problem are listed, with the reason, in the fixture's ``excluded`` table.

Run from a checkout, with VAMOS importable and pymoo installed (0.6.2 was used)::

    PYTHONPATH=src python tools/problem_reference_values.py --classpath "<jars>"

``--classpath`` lists, separated by the platform's path separator, the jmetal-core and
jmetal-problem 7.8 jars with commons-lang3 and commons-math3; ``tools/jmetal_values/JMetalValues.java``
runs with Java 17 or later. LIR-CMOP, DAS-CMOP, the classic constrained problems, CEC2009 CF1-CF10 and
ZXH-CF1-16 come from jMetal's ``main`` at commit ``JMETAL_MAIN_COMMIT``: 7.8 predates LIR-CMOP's
correction (jMetal #522), has no CEC2009 CF problems and fails to evaluate its ZXH-CF1.
``--classpath-main`` gives the same jars built from that commit. ``--only KEY ...`` regenerates those
problems and keeps the rest of the fixture, so the other sources are not needed::

    PYTHONPATH=src python tools/problem_reference_values.py --classpath-main "<jars>" --only lircmop1 dascmop1

Constraints are stored in the VAMOS convention, ``g <= 0`` feasible: pymoo uses it too, and jMetal's values
(``>= 0`` feasible) are negated. ZXH-CF9-16 are the exception: see ``DISJUNCTIVE_HEAD``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "foundation" / "data" / "problem_reference_values.json"
JAVA_HELPER = ROOT / "tools" / "jmetal_values" / "JMetalValues.java"
N_POINTS = 3

# VAMOS key -> pymoo name and the VAMOS attributes passed as keyword arguments (with the same name,
# except WFG's position parameter ``k``).
PYMOO: dict[str, tuple[str, tuple[str, ...]]] = {
    **{f"zdt{i}": (f"zdt{i}", ("n_var",)) for i in (1, 2, 3, 4, 6)},
    "zdt5": ("zdt5", ()),
    **{f"dtlz{i}": (f"dtlz{i}", ("n_var", "n_obj")) for i in range(1, 8)},
    **{f"wfg{i}": (f"wfg{i}", ("n_var", "n_obj", "k")) for i in range(1, 10)},
    **{f"mw{i}": (f"mw{i}", ("n_var",)) for i in range(1, 15)},
    **{
        key: (key, ("n_var", "n_obj"))
        for key in ("c1dtlz1", "c1dtlz3", "c2dtlz2", "c3dtlz1", "dc1dtlz1", "dc1dtlz3", "dc2dtlz1", "dc2dtlz3", "dc3dtlz1", "dc3dtlz3")
    },
}
# VAMOS key -> jMetal class (under org.uma.jmetal.problem.multiobjective), its constructor arguments
# and the number of variables of the VAMOS instance (None: the VAMOS default).
JMETAL: dict[str, tuple[str, str]] = {
    **{f"cec2009_uf{i}": (f"uf.UF{i}", "") for i in range(1, 11)},
    **{f"lz09_f{i}": (f"lz09.LZ09F{i}", "") for i in range(1, 10)},
    **{f"lsmop{i}": (f"lsmop.LSMOP{i}", "5,300,3") for i in range(5, 9)},
    **{f"zcat{i}": (f"zcat.ZCAT{i}", "2,30,false,1,false,false") for i in range(1, 21)},
    **{f"re{n}": (f"re.RE{n}", "") for n in (21, 22, 23, 24, 25, 31, 32, 33, 34, 35, 36, 37, 42, 61)},
    **{f"rwa{i}": (f"rwa.RWA{i}", "") for i in (2, 3, 4, 5, 7, 8)},
}
# The same, for problems that jMetal's main branch defines as VAMOS does: LIR-CMOP, DAS-CMOP, C3-DTLZ4,
# the classic constrained problems, the CEC2009 CF problems and ZXH-CF.
JMETAL_MAIN_COMMIT = "1e4b83cae6"
JMETAL_MAIN: dict[str, tuple[str, str]] = {
    **{f"lircmop{k}": (f"lircmop.LIRCMOP{k}", "") for k in range(1, 15)},
    **{f"dascmop{k}": (f"dascmop.DASCMOP{k}", "") for k in range(1, 10)},
    "c3dtlz4": ("cdtlz.C3_DTLZ4", "12,3,3"),
    "binh2": ("Binh2", ""),
    "constrex": ("ConstrEx", ""),
    "srinivas": ("Srinivas", ""),
    "tanaka": ("Tanaka", ""),
    "osyczka2": ("Osyczka2", ""),
    "golinski": ("Golinski", ""),
    "water": ("Water", ""),
    "car_side_impact": ("CarSideImpact", ""),
    **{f"cec2009_cf{k}": (f"cec2009.CEC2009CF{k}", "") for k in range(1, 11)},
    **{f"zxh_cf{k}": (f"cf.CF{k}", "") for k in range(1, 17)},
}
# ZXH-CF9-16 keep angles out of intervals. jMetal stores both bounds of each interval, ``<= 0`` feasible and
# not negated, after the first one or two constraints (two when the radius r > 0), and precomputes the
# violation: the smaller bound of an interval when both are positive. VAMOS stores that smaller value.
DISJUNCTIVE_HEAD: dict[str, int] = {f"zxh_cf{k}": 1 if k % 4 == 1 else 2 for k in range(9, 17)}
# DC-DTLZ with jMetal's constants (a = 3, b = 0.5, g unscaled in the constraints), which STEP also uses:
# values of jMetal's main at three points for each variant and dimension. jMetal puts DC3's constraint on g
# last; VAMOS, like pymoo, first.
DCDTLZ_JMETAL_OUTPUT = ROOT / "tests" / "foundation" / "data" / "dcdtlz_jmetal_values.json"
DCDTLZ_JMETAL_CASES = [
    (key, n_var, n_obj)
    for key in ("dc1dtlz1", "dc1dtlz3", "dc2dtlz1", "dc2dtlz3", "dc3dtlz1", "dc3dtlz3")
    for n_var, n_obj in ((12, 3), (12, 2), (2, 3))
]
# Variables that VAMOS encodes as an index into a table of values (the table's attribute): the
# reference evaluates the decoded value. Both problems document the encoding.
DECODED: dict[str, tuple[int, str]] = {"re22": (0, "_X1_VALUES"), "re25": (2, "_X3_VALUES")}
_LSMOP_LINKAGE = (
    "jMetal 7.8 applies the nonlinear variable linkage to every LSMOP problem (and LSMOP9 does not average "
    "its g); VAMOS follows the authors' PlatEMO, documented in vamos.foundation.problem.lsmop."
)
_RWA_C_FACTOR = (
    "The authors' C code (evo-mx/MO-RWA) and jMetal 7.8 write coefficients as '4.148 * 10e-3', ten times the "
    "paper's 4.148e-3; VAMOS uses the paper's values (VR-332)."
)
# Problems whose counterpart is not the same problem; the fixture records why.
EXCLUDED = {
    **{f"lsmop{i}": _LSMOP_LINKAGE for i in (1, 2, 3, 4, 9)},
    "rwa1": _RWA_C_FACTOR,
    "rwa6": _RWA_C_FACTOR,
    "rwa9": (
        "The authors' C code and jMetal 7.8 define RWA9 with 6 variables and compute a2, b2 and d2 as products of them "
        "(a2 = l1 * w1 * l2 * w2, about 2500, where the regression uses a2 - 11); VAMOS has the 10 variables (VR-331)."
    ),
    "rwa10": "The authors' C code and jMetal 7.8 add a constant to each objective, which the paper's Eq. 11 and VAMOS omit (VR-332).",
    "re41": (
        "jMetal 7.8's RE41 differs from the authors' reproblem.py (ryojitanabe/reproblems), which VAMOS follows: "
        "at the first fixture point the authors' f1 is 32.0024, VAMOS's 32.0024 and jMetal's 29.0500."
    ),
    "re91": (
        "jMetal 7.8's RE91 bounds x5 by 2.265 and differs from the authors' formulation at the mean of its "
        "random parameters, which VAMOS evaluates by default (bound 2.625)."
    ),
}


def _problem(key: str) -> Any:
    from vamos.foundation.problem.registry import make_problem_selection

    return make_problem_selection(key).instantiate()


def _reference_design(key: str, problem: Any, X: np.ndarray) -> np.ndarray:
    """``X`` with the index-encoded variable of ``key`` replaced by the value it selects."""
    if key not in DECODED:
        return X
    column, table = DECODED[key]
    values = np.asarray(getattr(problem, table), dtype=float)
    design = X.copy()
    design[:, column] = values[np.clip(np.rint(X[:, column]), 0, values.size - 1).astype(int)]
    return design


def _points(key: str, problem: Any) -> np.ndarray:
    """Three points drawn uniformly inside the VAMOS bounds (bits for a binary problem), seeded by the key."""
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    n_var = int(problem.n_var)
    if getattr(problem, "encoding", None) == "binary":
        return rng.integers(0, 2, size=(N_POINTS, n_var)).astype(float)
    lower = np.broadcast_to(np.asarray(problem.xl, dtype=float), (n_var,))
    upper = np.broadcast_to(np.asarray(problem.xu, dtype=float), (n_var,))
    return lower + rng.random((N_POINTS, n_var)) * (upper - lower)


def _pymoo_values(name: str, kwargs: dict[str, Any], X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    from pymoo.problems import get_problem

    reference = get_problem(name, **kwargs)
    if reference.n_var != X.shape[1]:
        raise ValueError(f"pymoo {name} has {reference.n_var} variables, VAMOS {X.shape[1]}")
    F, G = reference.evaluate(X, return_values_of=["F", "G"])
    G = None if G is None or np.size(G) == 0 else np.asarray(G, dtype=float).reshape(X.shape[0], -1)
    return np.asarray(F, dtype=float), G


def _jmetal_values(classpath: str, requests: list[tuple[str, str, np.ndarray]]) -> list[tuple[np.ndarray, np.ndarray | None]]:
    lines = []
    for class_name, arguments, x in requests:
        lines.append(f"org.uma.jmetal.problem.multiobjective.{class_name}\t{arguments}\t{','.join(repr(float(v)) for v in x)}")
    completed = subprocess.run(
        ["java", "-cp", classpath, str(JAVA_HELPER)],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"JMetalValues.java failed:\n{completed.stderr}")
    values: list[tuple[np.ndarray, np.ndarray | None]] = []
    for line in str(completed.stdout).splitlines():
        objectives, constraints = line.split("\t")
        F = np.array([float(v) for v in objectives.split(",")])
        G = -np.array([float(v) for v in constraints.split(",")]) if constraints else None
        values.append((F, G))
    return values


def _vamos_constraints(key: str, G: np.ndarray) -> np.ndarray:
    """jMetal's constraint values, negated, in VAMOS's columns: one per excluded interval for ZXH-CF9-16."""
    if key not in DISJUNCTIVE_HEAD:
        return G
    head, raw = DISJUNCTIVE_HEAD[key], -G
    return np.column_stack([raw[:, :head], raw[:, head:].reshape(raw.shape[0], -1, 2).min(axis=2)])


def _entry(problem: Any, source: str, implementation: str, X: np.ndarray, F: np.ndarray, G: np.ndarray | None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "source": source,
        "implementation": implementation,
        "n_var": int(problem.n_var),
        "n_obj": int(problem.n_obj),
        "X": X.tolist(),
        "F": F.tolist(),
    }
    if G is not None:
        entry["G"] = G.tolist()
    return entry


def _jmetal_entries(classpath: str, table: dict[str, tuple[str, str]], keys: list[str], source: str) -> dict[str, Any]:
    requests = []
    for key in keys:
        class_name, arguments = table[key]
        problem = _problem(key)
        requests.extend((class_name, arguments, x) for x in _reference_design(key, problem, _points(key, problem)))
    values = iter(_jmetal_values(classpath, requests))
    entries: dict[str, Any] = {}
    for key in keys:
        class_name, arguments = table[key]
        problem = _problem(key)
        X = _points(key, problem)
        rows = [next(values) for _ in range(N_POINTS)]
        F = np.vstack([row[0] for row in rows])
        constraints = [row[1] for row in rows]
        G = None if any(c is None for c in constraints) else _vamos_constraints(key, np.vstack([c for c in constraints if c is not None]))
        implementation = f"org.uma.jmetal.problem.multiobjective.{class_name}({arguments})"
        entries[key] = _entry(problem, source, implementation, X, F, G)
    return entries


def generate(
    classpath: str | None,
    classpath_main: str | None = None,
    only: list[str] | None = None,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The fixture: every problem, or only the keys of ``only`` on top of the ``existing`` fixture."""
    keys = set(only) if only is not None else {*PYMOO, *JMETAL, *JMETAL_MAIN}
    unknown = keys - {*PYMOO, *JMETAL, *JMETAL_MAIN}
    if unknown:
        raise ValueError(f"no reference implementation for {sorted(unknown)}")
    problems: dict[str, Any] = dict(existing["problems"]) if only is not None and existing is not None else {}

    pymoo_keys = [key for key in PYMOO if key in keys]
    if pymoo_keys:
        import pymoo

        for key in pymoo_keys:
            name, attributes = PYMOO[key]
            problem = _problem(key)
            kwargs: dict[str, Any] = {attribute: int(getattr(problem, attribute)) for attribute in attributes}
            if key == "zdt5":
                # pymoo normalizes ZDT5's objectives unless told otherwise; VAMOS and the paper do not.
                kwargs["normalize"] = False
            X = _points(key, problem)
            F, G = _pymoo_values(name, kwargs, X)
            problems[key] = _entry(problem, f"pymoo {pymoo.__version__}", f"pymoo.problems.get_problem({name!r}, **{kwargs})", X, F, G)

    for table, path, source in (
        (JMETAL, classpath, "jMetal 7.8"),
        (JMETAL_MAIN, classpath_main, f"jMetal main {JMETAL_MAIN_COMMIT}"),
    ):
        table_keys = [key for key in table if key in keys]
        if not table_keys:
            continue
        if path is None:
            raise ValueError(f"{source} values for {table_keys[0]} and others need its classpath")
        problems.update(_jmetal_entries(path, table, table_keys, source))

    return {
        "schema": "vamos.test.problem-reference-values/1",
        "generated_by": "tools/problem_reference_values.py",
        "constraint_convention": "g_lte_0",
        "points_per_problem": N_POINTS,
        "excluded": EXCLUDED,
        "problems": dict(sorted(problems.items())),
    }


def dcdtlz_jmetal_values(classpath_main: str) -> dict[str, Any]:
    """The fixture of DC-DTLZ with jMetal's constants: jMetal's values in VAMOS's constraint order."""
    requests, designs = [], []
    for key, n_var, n_obj in DCDTLZ_JMETAL_CASES:
        rng = np.random.default_rng(zlib.crc32(f"{key}-{n_var}-{n_obj}".encode()))
        X = rng.random((N_POINTS, n_var))
        designs.append(X)
        class_name = "dcdtlz." + key[:3].upper() + "_DTLZ" + key[-1]
        requests.extend((class_name, f"{n_var},{n_obj}", x) for x in X)
    values = iter(_jmetal_values(classpath_main, requests))
    cases = []
    for (key, n_var, n_obj), X in zip(DCDTLZ_JMETAL_CASES, designs):
        rows = [next(values) for _ in range(N_POINTS)]
        F = np.vstack([row[0] for row in rows])
        G = np.vstack([row[1] for row in rows if row[1] is not None])
        if key.startswith("dc3"):
            G = np.column_stack([G[:, -1], G[:, :-1]])
        cases.append({"key": key, "n_var": n_var, "n_obj": n_obj, "X": X.tolist(), "F": F.tolist(), "G": G.tolist()})
    return {
        "schema": "vamos.test.dcdtlz-jmetal-values/1",
        "generated_by": "tools/problem_reference_values.py --dcdtlz-jmetal",
        "source": f"jMetal main {JMETAL_MAIN_COMMIT}",
        "parameters": {"a": 3.0, "b": 0.5, "g_factor": 100.0, "constraint_g_scale": 1.0},
        "constraint_convention": "g_lte_0",
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--classpath", help="jmetal-core, jmetal-problem 7.8, commons-lang3 and commons-math3 jars")
    parser.add_argument("--classpath-main", help=f"the same jars built from jMetal's main at commit {JMETAL_MAIN_COMMIT}")
    parser.add_argument("--only", nargs="+", metavar="KEY", help="regenerate these problems and keep the rest of the fixture")
    parser.add_argument("--dcdtlz-jmetal", action="store_true", help=f"write {DCDTLZ_JMETAL_OUTPUT.name} instead")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if args.dcdtlz_jmetal:
        if args.classpath_main is None:
            parser.error("--dcdtlz-jmetal needs --classpath-main")
        DCDTLZ_JMETAL_OUTPUT.write_text(json.dumps(dcdtlz_jmetal_values(args.classpath_main), indent=1) + "\n", encoding="utf-8")
        print(f"wrote {DCDTLZ_JMETAL_OUTPUT}")
        return 0
    existing = json.loads(args.output.read_text(encoding="utf-8")) if args.only and args.output.is_file() else None
    fixture = generate(args.classpath, args.classpath_main, only=args.only, existing=existing)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(fixture['problems'])} problems to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
