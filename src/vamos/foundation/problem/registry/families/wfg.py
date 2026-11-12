from __future__ import annotations

from collections.abc import Callable

from ...wfg import (
    WFG1Problem,
    WFG2Problem,
    WFG3Problem,
    WFG4Problem,
    WFG5Problem,
    WFG6Problem,
    WFG7Problem,
    WFG8Problem,
    WFG9Problem,
)
from ..common import ProblemSpec


def _wfg_factory(cls: Callable[..., object], n_var: int, n_obj: int | None) -> object:
    return cls(n_var=n_var, n_obj=n_obj if n_obj is not None else 3)


SPECS: dict[str, ProblemSpec] = {}


def get_specs() -> dict[str, ProblemSpec]:
    if SPECS:
        return SPECS
    SPECS.update(
        {
            "wfg1": ProblemSpec(
                key="wfg1",
                label="WFG1",
                default_n_var=24,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="WFG1 benchmarking problem.",
                factory=lambda n_var, n_obj: _wfg_factory(WFG1Problem, n_var, n_obj),
            ),
            "wfg2": ProblemSpec(
                key="wfg2",
                label="WFG2",
                default_n_var=24,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="WFG2 benchmarking problem.",
                factory=lambda n_var, n_obj: _wfg_factory(WFG2Problem, n_var, n_obj),
            ),
            "wfg3": ProblemSpec(
                key="wfg3",
                label="WFG3",
                default_n_var=24,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="WFG3 benchmarking problem.",
                factory=lambda n_var, n_obj: _wfg_factory(WFG3Problem, n_var, n_obj),
            ),
            "wfg4": ProblemSpec(
                key="wfg4",
                label="WFG4",
                default_n_var=24,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="WFG4 benchmarking problem.",
                factory=lambda n_var, n_obj: _wfg_factory(WFG4Problem, n_var, n_obj),
            ),
            "wfg5": ProblemSpec(
                key="wfg5",
                label="WFG5",
                default_n_var=24,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="WFG5 benchmarking problem.",
                factory=lambda n_var, n_obj: _wfg_factory(WFG5Problem, n_var, n_obj),
            ),
            "wfg6": ProblemSpec(
                key="wfg6",
                label="WFG6",
                default_n_var=24,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="WFG6 benchmarking problem.",
                factory=lambda n_var, n_obj: _wfg_factory(WFG6Problem, n_var, n_obj),
            ),
            "wfg7": ProblemSpec(
                key="wfg7",
                label="WFG7",
                default_n_var=24,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="WFG7 benchmarking problem.",
                factory=lambda n_var, n_obj: _wfg_factory(WFG7Problem, n_var, n_obj),
            ),
            "wfg8": ProblemSpec(
                key="wfg8",
                label="WFG8",
                default_n_var=24,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="WFG8 benchmarking problem.",
                factory=lambda n_var, n_obj: _wfg_factory(WFG8Problem, n_var, n_obj),
            ),
            "wfg9": ProblemSpec(
                key="wfg9",
                label="WFG9",
                default_n_var=24,
                default_n_obj=3,
                allow_n_obj_override=True,
                description="WFG9 benchmarking problem.",
                factory=lambda n_var, n_obj: _wfg_factory(WFG9Problem, n_var, n_obj),
            ),
        }
    )
    return SPECS


__all__ = ["SPECS", "get_specs"]
