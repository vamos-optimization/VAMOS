from __future__ import annotations

from ...cec import (
    CEC2009CF1Problem,
    CEC2009UF1Problem,
    CEC2009UF2Problem,
    CEC2009UF3Problem,
    CEC2009UF4Problem,
    CEC2009UF5Problem,
    CEC2009UF6Problem,
    CEC2009UF7Problem,
    CEC2009UF8Problem,
    CEC2009UF9Problem,
    CEC2009UF10Problem,
)
from ..common import ProblemSpec

SPECS: dict[str, ProblemSpec] = {}


def get_specs() -> dict[str, ProblemSpec]:
    if SPECS:
        return SPECS
    SPECS.update(
        {
            "cec2009_uf1": ProblemSpec(
                key="cec2009_uf1",
                label="CEC2009 UF1",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="CEC2009 unconstrained function UF1.",
                factory=lambda n_var, _n_obj: CEC2009UF1Problem(n_var=n_var),
            ),
            "cec2009_uf2": ProblemSpec(
                key="cec2009_uf2",
                label="CEC2009 UF2",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="CEC2009 unconstrained function UF2.",
                factory=lambda n_var, _n_obj: CEC2009UF2Problem(n_var=n_var),
            ),
            "cec2009_uf3": ProblemSpec(
                key="cec2009_uf3",
                label="CEC2009 UF3",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="CEC2009 unconstrained function UF3.",
                factory=lambda n_var, _n_obj: CEC2009UF3Problem(n_var=n_var),
            ),
            "cec2009_uf4": ProblemSpec(
                key="cec2009_uf4",
                label="CEC2009 UF4",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="CEC2009 unconstrained function UF4.",
                factory=lambda n_var, _n_obj: CEC2009UF4Problem(n_var=n_var),
            ),
            "cec2009_uf5": ProblemSpec(
                key="cec2009_uf5",
                label="CEC2009 UF5",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="CEC2009 unconstrained function UF5.",
                factory=lambda n_var, _n_obj: CEC2009UF5Problem(n_var=n_var),
            ),
            "cec2009_uf6": ProblemSpec(
                key="cec2009_uf6",
                label="CEC2009 UF6",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="CEC2009 unconstrained function UF6.",
                factory=lambda n_var, _n_obj: CEC2009UF6Problem(n_var=n_var),
            ),
            "cec2009_uf7": ProblemSpec(
                key="cec2009_uf7",
                label="CEC2009 UF7",
                default_n_var=30,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="CEC2009 unconstrained function UF7.",
                factory=lambda n_var, _n_obj: CEC2009UF7Problem(n_var=n_var),
            ),
            "cec2009_uf8": ProblemSpec(
                key="cec2009_uf8",
                label="CEC2009 UF8",
                default_n_var=30,
                default_n_obj=3,
                allow_n_obj_override=False,
                description="CEC2009 unconstrained function UF8 (three objectives).",
                factory=lambda n_var, _n_obj: CEC2009UF8Problem(n_var=n_var),
            ),
            "cec2009_uf9": ProblemSpec(
                key="cec2009_uf9",
                label="CEC2009 UF9",
                default_n_var=30,
                default_n_obj=3,
                allow_n_obj_override=False,
                description="CEC2009 unconstrained function UF9 (three objectives).",
                factory=lambda n_var, _n_obj: CEC2009UF9Problem(n_var=n_var),
            ),
            "cec2009_uf10": ProblemSpec(
                key="cec2009_uf10",
                label="CEC2009 UF10",
                default_n_var=30,
                default_n_obj=3,
                allow_n_obj_override=False,
                description="CEC2009 unconstrained function UF10 (three objectives).",
                factory=lambda n_var, _n_obj: CEC2009UF10Problem(n_var=n_var),
            ),
            "cec2009_cf1": ProblemSpec(
                key="cec2009_cf1",
                label="CEC2009 CF1",
                default_n_var=10,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="CEC2009 constrained function CF1.",
                factory=lambda n_var, _n_obj: CEC2009CF1Problem(n_var=n_var),
            ),
        }
    )
    return SPECS


__all__ = ["SPECS", "get_specs"]
