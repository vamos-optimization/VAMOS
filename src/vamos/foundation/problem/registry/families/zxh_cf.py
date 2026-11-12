from __future__ import annotations

from ...zxh_cf import ZXHCFProblem
from ..common import ProblemSpec

SPECS: dict[str, ProblemSpec] = {}

_TYPES = ("A", "B", "C", "D")


def _n_var(n_obj: int) -> int:
    """M + 10 variables, as PlatEMO: jMetal's default 13 at three objectives."""
    return n_obj + 10


def _spec(k: int) -> ProblemSpec:
    return ProblemSpec(
        key=f"zxh_cf{k}",
        label=f"ZXH-CF{k}",
        default_n_var=13,
        default_n_obj=3,
        allow_n_obj_override=True,
        default_n_var_fn=_n_var,
        description=f"ZXH-CF{k}, a constrained problem of type {_TYPES[(k - 1) // 4]} (Zhou, Xiang and He, 2021).",
        factory=lambda n_var, n_obj: ZXHCFProblem(k, n_var=n_var, n_obj=3 if n_obj is None else n_obj),
    )


def get_specs() -> dict[str, ProblemSpec]:
    if SPECS:
        return SPECS
    SPECS.update({f"zxh_cf{k}": _spec(k) for k in range(1, 17)})
    return SPECS


__all__ = ["get_specs"]
