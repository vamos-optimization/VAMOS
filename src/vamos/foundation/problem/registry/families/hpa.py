from __future__ import annotations

from ...hpa.problems import HPA_INDICES, HPA_TYPES, HPAProblem
from ..common import ProblemSpec

SPECS: dict[str, ProblemSpec] = {}

_NAMES = {
    "power": "power",
    "drag": "drag",
    "velocity": "cruise speed",
    "deflection": "wing deflection",
    "twist": "wing twist",
    "efficiency": "span efficiency",
    "empty_weight": "empty weight",
    "span": "span",
    "root_aoa": "root angle of attack",
    "wire_tension": "wire tension",
    "payload": "payload",
}

# The constrained index ``MNL`` of each penalized index ``M0L``.
_CONSTRAINED = {base: index for index, (base, constrained) in HPA_INDICES.items() if constrained}


def _spec(index: str, level: int) -> ProblemSpec:
    base, constrained = HPA_INDICES[index]
    kind = HPA_TYPES[base]
    problem = HPAProblem(index, level)
    names = [_NAMES[name] for name in kind.objectives]
    objectives = f"{', '.join(names[:-1])} and {names[-1]}"
    handling = (
        f"{problem.n_constraints} constraints"
        if constrained
        else f"the constraints of HPA{_CONSTRAINED[base]} as penalties on the objectives"
    )
    return ProblemSpec(
        key=f"hpa{index}_{level}",
        label=f"HPA{index}-{level} (human-powered aircraft{'' if constrained else ', penalized'})",
        default_n_var=problem.n_var,
        default_n_obj=problem.n_obj,
        allow_n_obj_override=False,
        description=(
            f"Namura's HPA{index}-{level} (EMO 2025): human-powered aircraft wing design at difficulty level {level}, "
            f"minimizing {objectives}, with {handling}; the variables are normalized to [0, 1]."
        ),
        factory=lambda _n_var, _n_obj: HPAProblem(index, level),
        fixed_n_var=True,
    )


def get_specs() -> dict[str, ProblemSpec]:
    if SPECS:
        return SPECS
    for index in HPA_INDICES:
        for level in (0, 1, 2):
            spec = _spec(index, level)
            SPECS[spec.key] = spec
    return SPECS


__all__ = ["get_specs"]
