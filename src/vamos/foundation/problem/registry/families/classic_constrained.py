from __future__ import annotations

from ...classic_constrained import CLASSICS, ClassicConstrainedProblem
from ..common import ProblemSpec

SPECS: dict[str, ProblemSpec] = {}

_LABELS = {
    "binh2": ("Binh2", "Binh and Korn's second problem"),
    "constrex": ("ConstrEx", "Deb's constrained example (ConstrEx)"),
    "srinivas": ("Srinivas", "Srinivas and Deb's problem"),
    "tanaka": ("Tanaka", "Tanaka's problem"),
    "osyczka2": ("Osyczka2", "Osyczka and Kundu's problem"),
    "golinski": ("Golinski", "Golinski's speed reducer"),
    "water": ("Water", "the water resource planning problem"),
    "car_side_impact": ("Car side impact", "the car side impact problem"),
}


def _spec(key: str) -> ProblemSpec:
    _, lower, _, n_obj, n_constraints = CLASSICS[key]
    label, name = _LABELS[key]
    return ProblemSpec(
        key=key,
        label=label,
        default_n_var=len(lower),
        default_n_obj=n_obj,
        allow_n_obj_override=False,
        description=f"{name[0].upper()}{name[1:]}, with {n_constraints} constraints, as jMetal defines it.",
        factory=lambda _n_var, _n_obj: ClassicConstrainedProblem(key),
        fixed_n_var=True,
    )


def get_specs() -> dict[str, ProblemSpec]:
    if SPECS:
        return SPECS
    SPECS.update({key: _spec(key) for key in _LABELS})
    return SPECS


__all__ = ["get_specs"]
