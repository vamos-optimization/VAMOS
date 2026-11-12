"""RCM01-RCM50: the CEC 2021 suite of real-world constrained multiobjective problems.

A. Kumar, G. Wu, M. Z. Ali, Q. Luo, R. Mallipeddi, P. N. Suganthan and S. Das, "A benchmark-suite of real-world
constrained multi-objective optimization problems and some baseline results", Swarm and Evolutionary
Computation 67, 100961 (2021), https://doi.org/10.1016/j.swevo.2021.100961.

The definitions are those of jMetal's ``org.uma.jmetal.problem.multiobjective.cec2021`` classes at commit
``1e4b83cae6``, ported from the suite's reference code (``CEC2021_func.m``). That includes jMetal pull
request #514: RCM10's load ``P = 104``, and the constraint order of RCM26 and RCM28 (inequalities first). As in
jMetal:

- each equality constraint ``h(x) = 0`` (RCM22-RCM24, RCM26, RCM28 and RCM36-RCM50) is relaxed to
  ``|h(x)| <= 1e-4``;
- the integer and binary variables of RCM01, RCM06, RCM13, RCM15, RCM19, RCM25, RCM26, RCM28 and RCM29 are
  rounded inside the evaluation with Java's ``Math.round`` (halves up); the variables stay real;
- RCM09 has no constraints: the suite's metadata lists one, but its evaluator never sets it.

The groups are in :mod:`.mechanical_a` (RCM01-RCM11), :mod:`.mechanical_b` (RCM12-RCM21), :mod:`.process`
(RCM22-RCM29), :mod:`.power_electronics` (RCM30-RCM35) and :mod:`.power_system` (RCM36-RCM50). Constraints
follow VAMOS's convention, ``g <= 0`` feasible: they are jMetal's values with the opposite sign.
"""

from __future__ import annotations

import numpy as np

from vamos.foundation.problem.base import Problem
from vamos.foundation.problem.rcm import mechanical_a, mechanical_b, power_electronics, power_system, process
from vamos.foundation.problem.rcm._common import EQUALITY_TOLERANCE, Array, Entry

RCM: dict[str, Entry] = {
    **mechanical_a.PROBLEMS,
    **mechanical_b.PROBLEMS,
    **process.PROBLEMS,
    **power_electronics.PROBLEMS,
    **power_system.PROBLEMS,
}


class RCMProblem(Problem):
    """RCM``number`` (1-50), with jMetal's bounds; ``G`` is jMetal's constraints with the opposite sign."""

    def __init__(self, number: int) -> None:
        key = f"rcm{int(number):02d}"
        if key not in RCM:
            raise ValueError(f"Unknown RCM problem {number!r}; the problems are RCM01 to RCM50.")
        formula, lower, upper, n_obj, n_constraints = RCM[key]
        self.key = key
        self._formula = formula
        self.n_var = len(lower)
        self.n_obj = n_obj
        self.n_constraints = n_constraints
        self.xl = np.asarray(lower, dtype=float)
        self.xu = np.asarray(upper, dtype=float)

    def evaluate(self, X: Array, out: dict[str, Array]) -> None:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}).")
        f, c = self._formula(X.T)
        out["F"] = np.column_stack(f)
        if self.n_constraints:
            out["G"] = -np.column_stack(c)


__all__ = ["EQUALITY_TOLERANCE", "RCM", "RCMProblem"]
