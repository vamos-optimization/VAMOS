from __future__ import annotations

from typing import TYPE_CHECKING

from .common import ProblemFactory, ProblemSpec
from .families import (
    cec,
    classic_constrained,
    constrained_many,
    dtlz,
    lir_das_cmop,
    lsmop,
    lz09,
    misc,
    real_world,
    tanabe_ishibuchi_re,
    wfg,
    zapotecas_rwa,
    zcat,
    zdt,
    zxh_cf,
)

_PROBLEM_SPECS: dict[str, ProblemSpec] | None = None

if TYPE_CHECKING:
    PROBLEM_SPECS: dict[str, ProblemSpec]


def _build_problem_specs() -> dict[str, ProblemSpec]:
    specs: dict[str, ProblemSpec] = {}
    for family in (
        zdt.get_specs(),
        dtlz.get_specs(),
        lz09.get_specs(),
        cec.get_specs(),
        lsmop.get_specs(),
        constrained_many.get_specs(),
        classic_constrained.get_specs(),
        lir_das_cmop.get_specs(),
        zxh_cf.get_specs(),
        wfg.get_specs(),
        zcat.get_specs(),
        misc.get_specs(),
        real_world.get_specs(),
        tanabe_ishibuchi_re.get_specs(),
        zapotecas_rwa.get_specs(),
    ):
        specs.update(family)
    return specs


def get_problem_specs() -> dict[str, ProblemSpec]:
    global _PROBLEM_SPECS
    if _PROBLEM_SPECS is None:
        _PROBLEM_SPECS = _build_problem_specs()
    return _PROBLEM_SPECS


def available_problem_names() -> tuple[str, ...]:
    """Return the sorted names of all registered benchmark problems.

    Returns
    -------
    tuple[str, ...]
        Problem keys that can be passed to ``optimize()`` or
        ``make_problem_selection()``.
    """
    return tuple(get_problem_specs().keys())


def __getattr__(name: str) -> object:
    if name == "PROBLEM_SPECS":
        return get_problem_specs()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({"PROBLEM_SPECS", "get_problem_specs", "available_problem_names"} | set(globals()))


__all__ = ["ProblemSpec", "ProblemFactory", "PROBLEM_SPECS", "available_problem_names", "get_problem_specs"]
