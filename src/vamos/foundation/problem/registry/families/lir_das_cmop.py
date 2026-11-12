from __future__ import annotations

from ...dascmop import DASCMOPProblem
from ...lircmop import LIRCMOPProblem
from ..common import ProblemSpec

SPECS: dict[str, ProblemSpec] = {}


def _lircmop(k: int) -> ProblemSpec:
    n_obj = 3 if k >= 13 else 2
    return ProblemSpec(
        key=f"lircmop{k}",
        label=f"LIR-CMOP{k}",
        default_n_var=30,
        default_n_obj=n_obj,
        allow_n_obj_override=False,
        description=f"LIR-CMOP{k}, a constrained problem with large infeasible regions (Fan et al., 2019).",
        factory=lambda n_var, _n_obj: LIRCMOPProblem(k, n_var=n_var),
    )


def _dascmop(k: int) -> ProblemSpec:
    n_obj = 3 if k >= 7 else 2
    return ProblemSpec(
        key=f"dascmop{k}",
        label=f"DAS-CMOP{k}",
        default_n_var=30,
        default_n_obj=n_obj,
        allow_n_obj_override=False,
        description=f"DAS-CMOP{k} with jMetal's default difficulty (Fan et al., 2020).",
        factory=lambda n_var, _n_obj: DASCMOPProblem(k, n_var=n_var),
    )


def get_specs() -> dict[str, ProblemSpec]:
    if SPECS:
        return SPECS
    SPECS.update({f"lircmop{k}": _lircmop(k) for k in range(1, 15)})
    SPECS.update({f"dascmop{k}": _dascmop(k) for k in range(1, 10)})
    return SPECS


__all__ = ["get_specs"]
