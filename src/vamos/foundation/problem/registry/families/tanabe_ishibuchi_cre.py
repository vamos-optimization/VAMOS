from __future__ import annotations

from ...cre import CRE, CREProblem
from ..common import ProblemSpec

SPECS: dict[str, ProblemSpec] = {}

_LABELS = {
    "cre21": ("CRE21 (two-bar truss)", "two-bar truss design"),
    "cre22": ("CRE22 (welded beam)", "welded beam design"),
    "cre23": ("CRE23 (disc brake)", "disc brake design"),
    "cre24": ("CRE24 (speed reducer)", "speed reducer design, with its number of teeth rounded"),
    "cre25": ("CRE25 (gear train)", "gear train design, with its numbers of teeth rounded"),
    "cre31": ("CRE31 (car side impact)", "car side impact design"),
    "cre32": ("CRE32 (conceptual marine design)", "conceptual marine design"),
    "cre51": ("CRE51 (water resource planning)", "water resource planning"),
}


def _spec(key: str) -> ProblemSpec:
    _, lower, _, n_obj, n_constraints = CRE[key]
    label, name = _LABELS[key]
    return ProblemSpec(
        key=key,
        label=label,
        default_n_var=len(lower),
        default_n_obj=n_obj,
        allow_n_obj_override=False,
        description=f"Tanabe & Ishibuchi {key.upper()}: {name}, with {n_constraints} constraint{'s' if n_constraints > 1 else ''}, as jMetal defines it.",
        factory=lambda _n_var, _n_obj: CREProblem(key),
        fixed_n_var=True,
    )


def get_specs() -> dict[str, ProblemSpec]:
    if SPECS:
        return SPECS
    SPECS.update({key: _spec(key) for key in _LABELS})
    return SPECS


__all__ = ["get_specs"]
