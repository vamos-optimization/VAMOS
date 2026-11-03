"""Regenerate ``tests/reference/data/external_conformance.json``: pymoo's hypervolume on reference cases.

For each case, pymoo runs the same algorithm configuration as the VAMOS test in
``tests/reference/test_external_conformance.py`` over 30 seeds, and the fixture stores the mean and
standard deviation of the final hypervolume. The hypervolume is VAMOS's own, so only the algorithm
implementation differs. Run with VAMOS importable and pymoo installed (0.6.2 was used)::

    PYTHONPATH=src python tools/reference_conformance_baselines.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "reference" / "data" / "external_conformance.json"
SEEDS = list(range(30))

# The configuration of each case, shared with the VAMOS test.
CASES: dict[str, dict[str, Any]] = {
    "nsgaii-zdt1": {
        "problem": "zdt1",
        "n_var": 30,
        "n_obj": 2,
        "pop_size": 100,
        "max_evaluations": 10000,
        "crossover": {"name": "sbx", "prob": 0.9, "eta": 20.0},
        "mutation": {"name": "pm", "prob": "1/n", "eta": 20.0},
        "reference_point": [1.1, 1.1],
    },
    "nsgaiii-dtlz2": {
        "problem": "dtlz2",
        "n_var": 12,
        "n_obj": 3,
        "pop_size": 91,
        "divisions": 12,
        "max_evaluations": 91 * 110,
        "crossover": {"name": "sbx", "prob": 1.0, "eta": 30.0},
        "mutation": {"name": "pm", "prob": "1/n", "eta": 20.0},
        "reference_point": [1.1, 1.1, 1.1],
    },
}


def _pymoo_front(case: dict[str, Any], seed: int) -> np.ndarray:
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.optimize import minimize
    from pymoo.problems import get_problem
    from pymoo.util.ref_dirs import get_reference_directions

    problem = get_problem(case["problem"], n_var=case["n_var"], **({"n_obj": case["n_obj"]} if case["n_obj"] > 2 else {}))
    crossover = SBX(prob=case["crossover"]["prob"], eta=case["crossover"]["eta"])
    # VAMOS mutates every offspring, each variable with probability 1/n; pymoo's PM also gates each
    # individual with prob=0.9 unless told otherwise.
    mutation = PM(prob=1.0, prob_var=1.0 / case["n_var"], eta=case["mutation"]["eta"])
    # VAMOS, like Deb et al. and jMetal, keeps duplicates; pymoo removes them unless told otherwise.
    if "divisions" in case:
        directions = get_reference_directions("das-dennis", case["n_obj"], n_partitions=case["divisions"])
        algorithm = NSGA3(directions, pop_size=case["pop_size"], crossover=crossover, mutation=mutation, eliminate_duplicates=False)
    else:
        algorithm = NSGA2(pop_size=case["pop_size"], crossover=crossover, mutation=mutation, eliminate_duplicates=False)
        # VAMOS's binary tournament compares rank, then crowding; pymoo's default compares dominance first.
        algorithm.tournament_type = "comp_by_rank_and_crowding"
    result = minimize(problem, algorithm, ("n_evals", case["max_evaluations"]), seed=seed, verbose=False)
    return np.asarray(result.F, dtype=float)


def _hypervolume(F: np.ndarray, reference: list[float]) -> float:
    from vamos.foundation.quality_indicators.hypervolume import hypervolume

    point = np.asarray(reference, dtype=float)
    return float(hypervolume(F[np.all(F <= point, axis=1)], point))


def generate() -> dict[str, Any]:
    import pymoo

    baselines = {}
    for name, case in CASES.items():
        values = [_hypervolume(_pymoo_front(case, seed), case["reference_point"]) for seed in SEEDS]
        baselines[name] = {
            **case,
            "source": f"pymoo {pymoo.__version__}",
            "seeds": SEEDS,
            "hv_mean": round(float(np.mean(values)), 6),
            "hv_std": round(float(np.std(values, ddof=1)), 6),
        }
    return {"schema": "vamos.test.external-conformance/1", "generated_by": "tools/reference_conformance_baselines.py", "cases": baselines}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    fixture = generate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(fixture['cases'])} cases to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
