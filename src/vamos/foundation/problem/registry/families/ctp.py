from __future__ import annotations

from ...ctp import CTPProblem
from ..common import ProblemSpec

SPECS: dict[str, ProblemSpec] = {}


def _ctp(k: int) -> ProblemSpec:
    constraints = "two constraints" if k == 1 else "one constraint"
    return ProblemSpec(
        key=f"ctp{k}",
        label=f"CTP{k}",
        default_n_var=5,
        default_n_obj=2,
        allow_n_obj_override=False,
        description=f"CTP{k}, with {constraints} and the paper's parameters (Deb, Pratap and Meyarivan, 2001).",
        factory=lambda n_var, _n_obj: CTPProblem(k, n_var=n_var),
    )


def get_specs() -> dict[str, ProblemSpec]:
    if SPECS:
        return SPECS
    SPECS.update({f"ctp{k}": _ctp(k) for k in range(1, 8)})
    return SPECS


__all__ = ["get_specs"]
