"""VAMOS's hypervolume against pymoo's on the same configurations (VR-067).

``data/external_conformance.json`` holds pymoo 0.6.2's mean and standard deviation of the final
hypervolume over 30 seeds; ``tools/reference_conformance_baselines.py`` regenerates it, with the
configuration of each case. VAMOS runs the same configuration over 10 seeds, and its mean must lie
within 3 % of pymoo's. NSGA-II and NSGA-III are compared because pymoo implements them as VAMOS
does; its MOEA/D (no replacement limit), SMS-EMOA (tournament mating) and SPEA2 differ.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vamos import optimize
from vamos.engine.algorithm.config import NSGAIIConfig, NSGAIIIConfig
from vamos.foundation.problem.registry import make_problem_selection
from vamos.foundation.quality_indicators.hypervolume import hypervolume

_CASES: dict[str, Any] = json.loads((Path(__file__).parent / "data" / "external_conformance.json").read_text(encoding="utf-8"))["cases"]
_TOLERANCE = 0.03


def _config(case: dict[str, Any]) -> Any:
    builder = NSGAIIIConfig.builder() if "divisions" in case else NSGAIIConfig.builder()
    builder = builder.pop_size(case["pop_size"])
    builder = builder.crossover("sbx", prob=case["crossover"]["prob"], eta=case["crossover"]["eta"])
    builder = builder.mutation("pm", prob=case["mutation"]["prob"], eta=case["mutation"]["eta"])
    if "divisions" in case:
        # pymoo's NSGA-III mates random pairs (comp_by_cv_then_random); VAMOS's random selection does too.
        return builder.selection("random").reference_directions(divisions=case["divisions"]).build()
    return builder.selection("tournament", size=2).build()


@pytest.mark.reference
@pytest.mark.parametrize("name", sorted(_CASES))
def test_hypervolume_matches_pymoo(name: str) -> None:
    case = _CASES[name]
    problem = make_problem_selection(case["problem"], n_var=case["n_var"], n_obj=case["n_obj"]).instantiate()
    reference = np.asarray(case["reference_point"], dtype=float)
    values = []
    for seed in range(10):
        result = optimize(
            problem,
            algorithm=name.split("-")[0],
            algorithm_config=_config(case),
            max_evaluations=case["max_evaluations"],
            seed=seed,
            engine="numpy",
        )
        F = np.asarray(result.F, dtype=float)
        values.append(hypervolume(F[np.all(F <= reference, axis=1)], reference))

    mean = float(np.mean(values))
    assert abs(mean - case["hv_mean"]) <= _TOLERANCE * case["hv_mean"], (
        f"{name}: mean HV {mean:.4f} over 10 seeds; {case['source']} {case['hv_mean']:.4f} +- {case['hv_std']:.4f} over 30 seeds"
    )
