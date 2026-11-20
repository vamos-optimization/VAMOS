from __future__ import annotations

from ...modact import EVALUATOR_PACKAGES, MAXIMIZED, MODACT, N_VAR, OBJECTIVES, MODActProblem
from ..common import ProblemSpec

SPECS: dict[str, ProblemSpec] = {}


def _objectives(names: tuple[str, ...]) -> str:
    phrases = [f"{name} ({'maximized' if name in MAXIMIZED else 'minimized'})" for name in names]
    return f"{', '.join(phrases[:-1])} and {phrases[-1]}"


def _spec(name: str) -> ProblemSpec:
    n_obj, n_constraints = MODACT[name]
    objectives = OBJECTIVES[name.rstrip("1234")]
    return ProblemSpec(
        key=f"modact_{name}",
        label=f"MODAct {name.upper()} (actuator design)",
        default_n_var=N_VAR,
        default_n_obj=n_obj,
        allow_n_obj_override=False,
        description=(
            f"MODAct {name.upper()} (Picard & Schiffmann, 2021): actuator design, {_objectives(objectives)}, with "
            f"{n_constraints} constraints (level {name[-1]}); evaluated by the separately installed modact package "
            "(GPL-3.0), the modact extra."
        ),
        factory=lambda _n_var, _n_obj: MODActProblem(name),
        fixed_n_var=True,
        evaluator_packages=EVALUATOR_PACKAGES,
    )


def get_specs() -> dict[str, ProblemSpec]:
    if SPECS:
        return SPECS
    SPECS.update({f"modact_{name}": _spec(name) for name in MODACT})
    return SPECS


__all__ = ["get_specs"]
