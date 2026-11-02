from __future__ import annotations

from ...constrained_many import make_constrained_many_problem
from ..common import ProblemSpec


def _named(name: str, n_var: int, n_obj: int | None) -> object:
    return make_constrained_many_problem(name, n_var=n_var, n_obj=n_obj)


SPECS: dict[str, ProblemSpec] = {}


def get_specs() -> dict[str, ProblemSpec]:
    if SPECS:
        return SPECS

    SPECS.update(
        {
            "c1dtlz1": ProblemSpec(
                key="c1dtlz1",
                label="C1-DTLZ1",
                default_n_var=12,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Constrained DTLZ1 (C1-DTLZ1).",
                factory=lambda n_var, n_obj: _named("c1dtlz1", n_var, n_obj),
            ),
            "c1dtlz3": ProblemSpec(
                key="c1dtlz3",
                label="C1-DTLZ3",
                default_n_var=12,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Constrained DTLZ3 (C1-DTLZ3).",
                factory=lambda n_var, n_obj: _named("c1dtlz3", n_var, n_obj),
            ),
            "c2dtlz2": ProblemSpec(
                key="c2dtlz2",
                label="C2-DTLZ2",
                default_n_var=12,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Constrained DTLZ2 (C2-DTLZ2).",
                factory=lambda n_var, n_obj: _named("c2dtlz2", n_var, n_obj),
            ),
            "c3dtlz1": ProblemSpec(
                key="c3dtlz1",
                label="C3-DTLZ1",
                default_n_var=12,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Constrained DTLZ1 (C3-DTLZ1).",
                factory=lambda n_var, n_obj: _named("c3dtlz1", n_var, n_obj),
            ),
            "dc1dtlz1": ProblemSpec(
                key="dc1dtlz1",
                label="DC1-DTLZ1",
                default_n_var=12,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Discontinuous constrained DTLZ1 (DC1-DTLZ1).",
                factory=lambda n_var, n_obj: _named("dc1dtlz1", n_var, n_obj),
            ),
            "dc1dtlz3": ProblemSpec(
                key="dc1dtlz3",
                label="DC1-DTLZ3",
                default_n_var=12,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Discontinuous constrained DTLZ3 (DC1-DTLZ3).",
                factory=lambda n_var, n_obj: _named("dc1dtlz3", n_var, n_obj),
            ),
            "dc2dtlz1": ProblemSpec(
                key="dc2dtlz1",
                label="DC2-DTLZ1",
                default_n_var=12,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Discontinuous constrained DTLZ1 (DC2-DTLZ1).",
                factory=lambda n_var, n_obj: _named("dc2dtlz1", n_var, n_obj),
            ),
            "dc2dtlz3": ProblemSpec(
                key="dc2dtlz3",
                label="DC2-DTLZ3",
                default_n_var=12,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Discontinuous constrained DTLZ3 (DC2-DTLZ3).",
                factory=lambda n_var, n_obj: _named("dc2dtlz3", n_var, n_obj),
            ),
            "dc3dtlz1": ProblemSpec(
                key="dc3dtlz1",
                label="DC3-DTLZ1",
                default_n_var=12,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Discontinuous constrained DTLZ1 (DC3-DTLZ1).",
                factory=lambda n_var, n_obj: _named("dc3dtlz1", n_var, n_obj),
            ),
            "dc3dtlz3": ProblemSpec(
                key="dc3dtlz3",
                label="DC3-DTLZ3",
                default_n_var=12,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Discontinuous constrained DTLZ3 (DC3-DTLZ3).",
                factory=lambda n_var, n_obj: _named("dc3dtlz3", n_var, n_obj),
            ),
            "mw1": ProblemSpec(
                key="mw1",
                label="MW1",
                default_n_var=15,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="Constrained MW1 problem.",
                factory=lambda n_var, n_obj: _named("mw1", n_var, n_obj),
            ),
            "mw2": ProblemSpec(
                key="mw2",
                label="MW2",
                default_n_var=15,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="Constrained MW2 problem.",
                factory=lambda n_var, n_obj: _named("mw2", n_var, n_obj),
            ),
            "mw3": ProblemSpec(
                key="mw3",
                label="MW3",
                default_n_var=15,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="Constrained MW3 problem.",
                factory=lambda n_var, n_obj: _named("mw3", n_var, n_obj),
            ),
            "mw4": ProblemSpec(
                key="mw4",
                label="MW4",
                default_n_var=15,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Constrained MW4 problem.",
                factory=lambda n_var, n_obj: _named("mw4", n_var, n_obj),
            ),
            "mw5": ProblemSpec(
                key="mw5",
                label="MW5",
                default_n_var=15,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="Constrained MW5 problem.",
                factory=lambda n_var, n_obj: _named("mw5", n_var, n_obj),
            ),
            "mw6": ProblemSpec(
                key="mw6",
                label="MW6",
                default_n_var=15,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="Constrained MW6 problem.",
                factory=lambda n_var, n_obj: _named("mw6", n_var, n_obj),
            ),
            "mw7": ProblemSpec(
                key="mw7",
                label="MW7",
                default_n_var=15,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="Constrained MW7 problem.",
                factory=lambda n_var, n_obj: _named("mw7", n_var, n_obj),
            ),
            "mw8": ProblemSpec(
                key="mw8",
                label="MW8",
                default_n_var=15,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Constrained MW8 problem.",
                factory=lambda n_var, n_obj: _named("mw8", n_var, n_obj),
            ),
            "mw9": ProblemSpec(
                key="mw9",
                label="MW9",
                default_n_var=15,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="Constrained MW9 problem.",
                factory=lambda n_var, n_obj: _named("mw9", n_var, n_obj),
            ),
            "mw10": ProblemSpec(
                key="mw10",
                label="MW10",
                default_n_var=15,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="Constrained MW10 problem.",
                factory=lambda n_var, n_obj: _named("mw10", n_var, n_obj),
            ),
            "mw11": ProblemSpec(
                key="mw11",
                label="MW11",
                default_n_var=15,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="Constrained MW11 problem.",
                factory=lambda n_var, n_obj: _named("mw11", n_var, n_obj),
            ),
            "mw12": ProblemSpec(
                key="mw12",
                label="MW12",
                default_n_var=15,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="Constrained MW12 problem.",
                factory=lambda n_var, n_obj: _named("mw12", n_var, n_obj),
            ),
            "mw13": ProblemSpec(
                key="mw13",
                label="MW13",
                default_n_var=15,
                default_n_obj=2,
                allow_n_obj_override=False,
                description="Constrained MW13 problem.",
                factory=lambda n_var, n_obj: _named("mw13", n_var, n_obj),
            ),
            "mw14": ProblemSpec(
                key="mw14",
                label="MW14",
                default_n_var=15,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="Constrained MW14 problem.",
                factory=lambda n_var, n_obj: _named("mw14", n_var, n_obj),
            ),
        }
    )
    return SPECS


__all__ = ["SPECS", "get_specs"]
