"""Regenerate ``tests/foundation/data/rcm_reformulation_reference_values.json`` from the authors' decoder scripts.

The RCM reformulations of ``vamos.foundation.problem.rcm.reformulations`` port the decoders of the scripts
that computed the reference fronts of the CRE and RCM suites: ``process_and_sopwm.py`` (RCM22, RCM24, RCM28
and the sorted and level variants of RCM30-RCM35), ``dg_sizing.py`` (RCM36-RCM39), ``optimal_power_flow.py``
(RCM40-RCM46) and ``microgrid.py`` (RCM47-RCM49), in Evolver's ``scripts/realworld_repair`` at commit
``SOURCE_COMMIT``. For 20 fixed genotypes per decoder, the fixture holds the genotypes, whether each script
decodes them successfully (its design lies within the bounds, and for a power flow the solve converged) and the
decoded designs: every row of a closed-form decoder, the successful rows of a power flow.
``tests/foundation/test_rcm_reformulations.py`` checks VAMOS's decoders against them. The cumulative variant of
RCM30-RCM35 has no script; the test checks it against Wang et al.'s equations instead.

Run from a checkout, with NumPy and SciPy installed (the scripts' ``feasible_controls`` finds controls whose
power flow lies within the bounds with SciPy's L-BFGS-B)::

    python tools/rcm_reformulation_reference_values.py --scripts <evolver>/scripts/realworld_repair

The scripts are imported without writing bytecode, so their checkout stays unchanged.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "foundation" / "data" / "rcm_reformulation_reference_values.json"
SOURCE_COMMIT = "4a2612bb2"
SEED = 20261120
ROWS = 20


def _rows(rng: np.random.Generator, lower: np.ndarray, upper: np.ndarray, edges: list[np.ndarray]) -> np.ndarray:
    """``edges`` first, then uniform rows between the bounds, ROWS in all."""
    random = lower + rng.random((ROWS - len(edges), lower.size)) * (upper - lower)
    return np.vstack([*(np.broadcast_to(edge, lower.shape)[None, :] for edge in edges), random])


def _case(name: str, problems: list[str], variant: str | None, U: np.ndarray, X: np.ndarray, decoded: np.ndarray) -> dict[str, Any]:
    """A power flow's failed rows keep no design; a closed-form decoder's rows all do."""
    closed_form = name.startswith("process_and_sopwm.")
    return {
        "decoder": name,
        "problems": problems,
        "variant": variant,
        "U": U.tolist(),
        "decoded": decoded.tolist(),
        "X": [row.tolist() if ok or closed_form else None for row, ok in zip(X, decoded)],
    }


def _inside(process: ModuleType, k: int, X: np.ndarray) -> np.ndarray:
    lower, upper = process.BOUNDS[k]
    return np.asarray(((X >= lower) & (X <= upper)).all(axis=1))


def _cases(scripts: dict[str, ModuleType], rng: np.random.Generator) -> list[dict[str, Any]]:
    process, dg, opf, microgrid = scripts["process_and_sopwm"], scripts["dg_sizing"], scripts["optimal_power_flow"], scripts["microgrid"]
    cases: list[dict[str, Any]] = []
    snap = process.SNAP
    for k, n_genes, lower in ((22, 5, -snap), (24, 1, -snap), (28, 2, None)):
        low = np.full(n_genes, -snap)
        if lower is None:
            low[0] = 0.0
        high = np.full(n_genes, 1.0 + snap)
        if lower is None:
            high[0] = 1.0
        U = _rows(rng, low, high, [low, high, np.full(n_genes, 0.5), np.full(n_genes, 0.0), np.full(n_genes, 1.0)])
        X = getattr(process, f"decode_rcm{k}")(U)
        cases.append(_case(f"process_and_sopwm.decode_rcm{k}", [f"rcm{k}"], None, U, X, _inside(process, k, X)))
    for k in range(30, 36):
        d = len(process.BOUNDS[k][0])
        low, high = np.full(d, -4.5), np.full(d, 94.5)
        edges = [low, high, np.full(d, 45.0), np.linspace(-4.5, 94.5, d)[::-1], np.r_[np.full(d - 3, 90.0), [10.0, 20.0, 30.0]]]
        U = _rows(rng, low, high, edges)
        X = process.decode_sopwm(U)
        cases.append(_case("process_and_sopwm.decode_sopwm", [f"rcm{k}"], "sorted", U, X, _inside(process, k, X)))
        low, high = np.append(low, -snap), np.append(high, 1.0 + snap)
        edges = [low, high, np.append(np.full(d, 45.0), 0.5), np.append(np.full(d, 90.0), 1.0), np.append(np.full(d, 0.0), 0.0)]
        U = _rows(rng, low, high, edges)
        X = process.decode_sopwm_level(k, U)
        cases.append(_case("process_and_sopwm.decode_sopwm_level", [f"rcm{k}"], "level", U, X, _inside(process, k, X)))

    U = _rows(rng, np.zeros(4), np.ones(4), [np.zeros(4), np.ones(4), np.array([1.0, 0.0, 1.0, 0.0])])
    X, ok, _ = dg.decode(U)
    cases.append(_case("dg_sizing.decode", [f"rcm{k}" for k in range(36, 40)], None, U, X, ok))

    feasible = opf.feasible_controls(rng, ROWS // 2)
    U = np.vstack([feasible, rng.random((ROWS - feasible.shape[0], 8))])
    X, excess, converged = opf.decode(U)
    cases.append(_case("optimal_power_flow.decode", [f"rcm{k}" for k in range(40, 47)], None, U, X, converged & (excess == 0.0)))

    edges = [np.zeros(7), np.ones(7), np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5])]
    U = _rows(rng, np.zeros(7), np.ones(7), edges)
    X, hmax, converged, _ = microgrid.decode(U)
    _, ok = microgrid.feasible_objectives(47, X, hmax, converged)
    cases.append(_case("microgrid.decode", [f"rcm{k}" for k in range(47, 50)], None, U, X, ok))
    return cases


def reference_values(scripts_dir: Path) -> dict[str, Any]:
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(scripts_dir))
    try:
        scripts = {name: importlib.import_module(name) for name in ("process_and_sopwm", "dg_sizing", "optimal_power_flow", "microgrid")}
        cases = _cases(scripts, np.random.default_rng(SEED))
    finally:
        sys.path.remove(str(scripts_dir))
    return {
        "schema": "vamos.test.rcm-reformulation-reference-values/1",
        "generated_by": "tools/rcm_reformulation_reference_values.py",
        "source": f"Evolver scripts/realworld_repair, commit {SOURCE_COMMIT}",
        "seed": SEED,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scripts", type=Path, required=True, help="Evolver's scripts/realworld_repair directory")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    text = json.dumps(reference_values(args.scripts.resolve()), indent=1) + "\n"
    args.output.write_text(text, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
