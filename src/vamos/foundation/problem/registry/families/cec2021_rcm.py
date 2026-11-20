from __future__ import annotations

from collections.abc import Callable

from ...rcm import RCM, RCMProblem
from ...rcm.reformulations import REFORMULATED, SOPWMReformulation, make_reformulation
from ..common import ProblemSpec

SPECS: dict[str, ProblemSpec] = {}

# The name of each problem in the suite; "(rounded)" marks the ones whose integer variables jMetal rounds,
# "(equalities)" the ones with equality constraints relaxed to |h| <= 1e-4.
_NAMES = {
    1: "pressure vessel design (rounded)",
    2: "vibrating platform design",
    3: "two-bar truss design",
    4: "welded beam design",
    5: "disc brake design",
    6: "speed reducer design (rounded)",
    7: "gear train design",
    8: "car side impact design",
    9: "four-bar plane truss",
    10: "two-bar plane truss",
    11: "water resource management",
    12: "simply supported I-beam design",
    13: "gear box design (rounded)",
    14: "multiple-disk clutch brake design",
    15: "spring design (rounded)",
    16: "cantilever beam design",
    17: "bulk carrier design",
    18: "front rail design",
    19: "multi-product batch plant (rounded)",
    20: "hydrostatic thrust bearing design",
    21: "crash energy management for a high-speed train",
    22: "Haverly's pooling problem (equalities)",
    23: "reactor network design (equalities)",
    24: "heat exchanger network design (equalities)",
    25: "process synthesis (rounded)",
    26: "process synthesis and design (rounded, equalities)",
    27: "process flow sheeting",
    28: "two-reactor problem (rounded, equalities)",
    29: "process synthesis, second problem (rounded)",
    30: "SOPWM of 3-level inverters",
    31: "SOPWM of 5-level inverters",
    32: "SOPWM of 7-level inverters",
    33: "SOPWM of 9-level inverters",
    34: "SOPWM of 11-level inverters",
    35: "SOPWM of 13-level inverters",
    36: "distributed generation sizing: phase balance and active power loss (equalities)",
    37: "distributed generation sizing: phase balance and reactive power loss (equalities)",
    38: "distributed generation sizing: active and reactive power loss (equalities)",
    39: "distributed generation sizing: phase balance, active and reactive power loss (equalities)",
    40: "optimal power flow: active and reactive power loss (equalities)",
    41: "optimal power flow: voltage deviation, active and reactive power loss (equalities)",
    42: "optimal power flow: voltage deviation and active power loss (equalities)",
    43: "optimal power flow: fuel cost and active power loss (equalities)",
    44: "optimal power flow: fuel cost, active and reactive power loss (equalities)",
    45: "optimal power flow: fuel cost, voltage deviation and active power loss (equalities)",
    46: "optimal power flow: fuel cost, voltage deviation, active and reactive power loss (equalities)",
    47: "islanded microgrid droop control: active and reactive power loss (equalities)",
    48: "islanded microgrid droop control: active power loss and voltage deviation (equalities)",
    49: "islanded microgrid droop control: active and reactive power loss and voltage deviation (equalities)",
    50: "power distribution system planning (equalities)",
}


_NOTES = {"rounded": "integer variables rounded", "equalities": "equalities relaxed to |h| <= 1e-4"}


def _spec(k: int) -> ProblemSpec:
    key = f"rcm{k:02d}"
    _, lower, _, n_obj, n_constraints = RCM[key]
    name, _, notes = _NAMES[k].partition(" (")
    extra = "".join(f"; {_NOTES[note]}" for note in notes.rstrip(")").split(", ") if note)
    constraints = f"{n_constraints} constraint{'s' if n_constraints > 1 else ''}" if n_constraints else "no constraints"
    return ProblemSpec(
        key=key,
        label=f"RCM{k:02d} ({name})",
        default_n_var=len(lower),
        default_n_obj=n_obj,
        allow_n_obj_override=False,
        description=f"CEC 2021 RCM{k:02d} (Kumar et al., 2021): {name}, with {constraints}, as jMetal defines it{extra}.",
        factory=lambda _n_var, _n_obj: RCMProblem(k),
        fixed_n_var=True,
    )


# What each reformulation searches; the decoded design meets the original's equalities or ordering constraints.
_PROCESS_GENOTYPES = {
    22: "the pool quality, split and size and the feed C of both products",
    24: "the offset of x7 above 100, with x1, x2, x4 and x6 at 0",
    28: "the reactor used and its volume",
}


def _genotype(k: int) -> str:
    if k in _PROCESS_GENOTYPES:
        return _PROCESS_GENOTYPES[k]
    if k <= 35:
        return "cumulative shares of the angles (Wang et al., 2025); variant 'sorted' or 'level'"
    if k <= 39:
        return "the four generator sizes, the voltages solved by a Newton power flow"
    if k <= 46:
        return "the eight generator powers, the voltages solved by a Newton power flow"
    return "seven controls, the voltages and droop scale solved by a damped Newton method"


def _sopwm_factory(k: int) -> Callable[..., object]:
    default_n_var = SOPWMReformulation(k).n_var

    def factory(n_var: int, _n_obj: int | None, *, variant: str = "cumulative") -> SOPWMReformulation:
        problem = SOPWMReformulation(k, variant=variant)
        # The variant sets the number of variables: the key's default, or the variant's own number, is accepted.
        if n_var not in (default_n_var, problem.n_var):
            raise ValueError(f"RCM{k:02d}'s reformulation with variant {variant!r} has {problem.n_var} variables; got n_var={n_var}.")
        return problem

    return factory


def _reformulated_spec(k: int) -> ProblemSpec:
    problem = make_reformulation(k)
    name = _NAMES[k].partition(" (")[0]
    sopwm = 30 <= k <= 35
    return ProblemSpec(
        key=problem.key,
        label=f"RCM{k:02d} reformulated ({name})",
        default_n_var=problem.n_var,
        default_n_obj=problem.n_obj,
        allow_n_obj_override=False,
        description=(
            f"CEC 2021 RCM{k:02d} (Kumar et al., 2021) searched through {_genotype(k)}; "
            f"the objectives and inequality constraints are rcm{k:02d}'s at the decoded design, followed by a "
            "decoding-failure constraint."
        ),
        factory=_sopwm_factory(k) if sopwm else lambda _n_var, _n_obj: make_reformulation(k),
        # A SOPWM variant has its own number of variables, which the factory checks.
        fixed_n_var=not sopwm,
    )


def get_specs() -> dict[str, ProblemSpec]:
    if SPECS:
        return SPECS
    SPECS.update({f"rcm{k:02d}": _spec(k) for k in _NAMES})
    SPECS.update({f"rcm{k:02d}_reformulated": _reformulated_spec(k) for k in REFORMULATED})
    return SPECS


__all__ = ["get_specs"]
