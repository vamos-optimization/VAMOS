"""Rule G7 for every built-in grammar: its default configuration runs as its algorithm's ``default()``.

Each space runs its defaults and ``XConfig.default(n_var=..., n_obj=..., encoding=...)`` with the same
seed on a problem of its encoding and compares the final populations. The reference-direction
algorithms (MOEA/D, NSGA-III and RVEA), whose population depends on the number of objectives, also
run a three-objective problem.

The nine mixed spaces do not pass yet: their operators take a choice and a probability per segment
(``perm_mutation``, ``real_mutation_prob``, ...), while the default mixed operators apply one
probability to every segment and draw their random numbers in another order, so no point of those
spaces runs as the default. The strict ``xfail`` records the departure until the mixed grammars
offer the default operators.
"""

from __future__ import annotations

import pytest

from vamos.engine.tuning.grammar import builtin_parameter_space, builtin_parameter_space_names, check_default_configuration
from vamos.foundation.problem.registry import make_problem_selection

CONSTRAINED_ALGORITHMS = {"ccmo", "ctaea", "top", "cmoeams"}
PROBLEMS = {
    "real": ("zdt1", {"n_var": 10}),
    "constrained": ("cec2009_cf1", {}),
    "binary": ("bin_knapsack", {}),
    "integer": ("int_alloc", {}),
    "permutation": ("tsp6", {}),
    "mixed": ("mixed_design", {}),
}
# Every mixed space: each one runs its segment settings (test_mixed_segment_settings.py).
MIXED_DEPARTURES = {name for name in builtin_parameter_space_names() if name.endswith("_mixed")}
MIXED_DEPARTURE = pytest.mark.xfail(
    strict=True,
    reason="the mixed operators draw per-segment choices in another order than the default mixed operators",
)


def _problem_kind(name: str) -> str:
    space = builtin_parameter_space(name)
    if space.encoding != "real":
        return space.encoding
    return "constrained" if name.endswith("_constrained") or space.algorithm in CONSTRAINED_ALGORITHMS else "real"


def _cases() -> list[object]:
    cases: list[object] = []
    for name in sorted(builtin_parameter_space_names()):
        marks = [MIXED_DEPARTURE] if name in MIXED_DEPARTURES else []
        cases.append(pytest.param(name, None, id=name, marks=marks))
    for name in ("moead", "nsgaiii", "rvea"):
        cases.append(pytest.param(name, ("dtlz2", {"n_obj": 3, "n_var": 12}), id=f"{name}-three-objectives"))
    return cases


@pytest.mark.parametrize(("name", "problem"), _cases())
def test_the_default_configuration_runs_as_the_algorithms_default(name: str, problem: tuple[str, dict[str, int]] | None) -> None:
    key, kwargs = problem if problem is not None else PROBLEMS[_problem_kind(name)]
    instance = make_problem_selection(key, **kwargs).instantiate()

    check_default_configuration(builtin_parameter_space(name), instance, seed=7, max_evaluations=400)
