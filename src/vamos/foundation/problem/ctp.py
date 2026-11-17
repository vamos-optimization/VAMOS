"""CTP1-CTP7: constrained test problems with tunable difficulty.

K. Deb, A. Pratap and T. Meyarivan, "Constrained Test Problems for Multi-objective Evolutionary Optimization",
EMO 2001, LNCS 1993, 284-298, doi:10.1007/3-540-44719-9_20. Both objectives are minimized and ``f1 = x1``, with
``x1`` in [0, 1]. The paper writes each constraint as ``c(x) >= 0``; VAMOS stores ``G = -c``, so that ``G <= 0``
is feasible.

- CTP1 (the paper's eq. 4): ``f2 = g exp(-f1 / g)``, with J constraints ``c_j = f2 - a_j exp(-b_j f1) >= 0``.
  :func:`ctp1_parameters` computes ``a_j`` and ``b_j`` by the paper's procedure (its Steps 1-3).
- CTP2-CTP7 (eq. 5): ``f2 = g (1 - f1 / g)``, with one constraint
  ``c = cos(theta) (f2 - e) - sin(theta) f1 - a |sin(b pi (sin(theta) (f2 - e) + cos(theta) f1)^c)|^d >= 0``
  and the six parameters of :data:`PAPER_PARAMETERS`.

The paper's experiments use "Rastrigin's function as the g functional" with five variables, without giving its
formula. VAMOS uses the Rastrigin function of ZDT4 (Deb, 1999), ``g = 1 + 10 (n - 1) + sum_{i>=2} (x_i^2 -
10 cos(4 pi x_i))`` with ``x_i`` in [-5, 5], whose minimum is ``g = 1`` at ``x_i = 0``. It computes that function
as ``1 + sum_{i>=2} (x_i^2 + 20 sin^2(2 pi x_i))``, without cancelling ``10 (n - 1)`` against the cosines, and
CTP2-CTP7's ``f2`` as ``g - f1``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from vamos.foundation.problem.base import Problem

Array = np.ndarray[Any, Any]

# The paper's parameters (theta, a, b, c, d, e) of CTP2-CTP7, theta in radians.
PAPER_PARAMETERS: dict[int, tuple[float, float, float, int, float, float]] = {
    2: (-0.2 * math.pi, 0.2, 10.0, 1, 6.0, 1.0),
    3: (-0.2 * math.pi, 0.1, 10.0, 1, 0.5, 1.0),
    4: (-0.2 * math.pi, 0.75, 10.0, 1, 0.5, 1.0),
    5: (-0.2 * math.pi, 0.1, 10.0, 2, 0.5, 1.0),
    6: (0.1 * math.pi, 40.0, 0.5, 1, 2.0, -2.0),
    7: (-0.05 * math.pi, 40.0, 5.0, 1, 6.0, 0.0),
}
# CTP1's default number of constraints, J = 2 in the paper.
CTP1_CONSTRAINTS = 2


def ctp1_parameters(n_constraints: int = CTP1_CONSTRAINTS) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """CTP1's ``(a_1, ..., a_J)`` and ``(b_1, ..., b_J)`` for ``J = n_constraints``, by the paper's procedure.

    With ``a_0 = b_0 = 1`` and ``x_j = j / (J + 1)``, ``y = a_{j-1} exp(-b_{j-1} x_j)`` is the previous boundary
    at ``x_j``; ``a_j = (a_{j-1} + y) / 2`` and ``b_j = -ln(y / a_j) / x_j`` make the next boundary pass through
    ``(x_j, y)``. With ``J = 2``: ``a = (0.858266, 0.728234)`` and ``b = (0.541475, 0.295039)``, the paper's
    ``(0.858, 0.728)`` and ``(0.541, 0.295)`` before rounding.
    """
    count = _positive_integer("CTP1's n_constraints", n_constraints)
    a, b = [1.0], [1.0]
    for j in range(count):
        x = (j + 1) / (count + 1)
        y = a[j] * math.exp(-b[j] * x)
        a.append(0.5 * (a[j] + y))
        b.append(-math.log(y / a[j + 1]) / x)
    return tuple(a[1:]), tuple(b[1:])


def rastrigin_g(X: Array) -> Array:
    """The distance function ``g`` of every row of the full decision matrix ``X``, from its columns 2 to n."""
    rest = X[:, 1:]
    return np.asarray(1.0 + np.sum(rest * rest + 20.0 * np.sin(2.0 * np.pi * rest) ** 2, axis=1))


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if not float(value).is_integer() or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"CTP's {name} must be finite, got {value!r}.")
    return number


class CTPProblem(Problem):
    """CTP``variant`` (1-7) with ``n_var`` variables: ``x1`` in [0, 1] and the others in [-5, 5].

    CTP1 has ``n_constraints`` constraints (the paper's J, 2 by default), with the coefficients ``a_j`` and
    ``b_j`` of :func:`ctp1_parameters`. CTP2-CTP7 have one constraint, whose parameters ``theta`` (in radians),
    ``a``, ``b``, ``c``, ``d`` and ``e`` default to the paper's (:data:`PAPER_PARAMETERS`); ``c`` must be a
    positive integer, since ``sin(theta) (f2 - e) + cos(theta) f1`` can be negative. The attributes of the same
    names hold them: ``a_j`` and ``b_j`` (empty for CTP2-CTP7), and ``theta`` to ``e`` for CTP2-CTP7 only.
    Constraints follow VAMOS's convention, ``g <= 0`` feasible: ``G = -c``.
    """

    a_j: tuple[float, ...]
    b_j: tuple[float, ...]
    theta: float
    a: float
    b: float
    c: int
    d: float
    e: float

    def __init__(
        self,
        variant: int,
        *,
        n_var: int = 5,
        n_constraints: int | None = None,
        theta: float | None = None,
        a: float | None = None,
        b: float | None = None,
        c: int | None = None,
        d: float | None = None,
        e: float | None = None,
    ) -> None:
        number = int(variant)
        if number not in range(1, 8):
            raise ValueError(f"Unknown CTP variant {variant!r}; the variants are 1 to 7.")
        if isinstance(n_var, bool) or not isinstance(n_var, (int, np.integer)) or int(n_var) < 2:
            raise ValueError(f"CTP{number} needs an integer n_var of at least 2 (x1 and one variable of g), got {n_var!r}.")
        self.variant = number
        self.n_var = int(n_var)
        self.n_obj = 2
        self.xl = np.concatenate([[0.0], np.full(self.n_var - 1, -5.0)])
        self.xu = np.concatenate([[1.0], np.full(self.n_var - 1, 5.0)])
        generator = {"theta": theta, "a": a, "b": b, "c": c, "d": d, "e": e}
        if number == 1:
            given = [name for name, value in generator.items() if value is not None]
            if given:
                raise ValueError(f"CTP1 has no parameter {', '.join(given)}; theta, a, b, c, d and e belong to CTP2-CTP7.")
            self.a_j, self.b_j = ctp1_parameters(CTP1_CONSTRAINTS if n_constraints is None else n_constraints)
            self.n_constraints = len(self.a_j)
            return
        if n_constraints is not None and n_constraints != 1:
            raise ValueError(f"CTP{number} has one constraint; n_constraints sets CTP1's J, got {n_constraints!r}.")
        values = [default if value is None else value for value, default in zip(generator.values(), PAPER_PARAMETERS[number])]
        self.theta = _finite("theta", values[0])
        self.a = _finite("a", values[1])
        self.b = _finite("b", values[2])
        self.c = _positive_integer("CTP's c", values[3])
        self.d = _finite("d", values[4])
        self.e = _finite("e", values[5])
        if self.a < 0.0 or self.d <= 0.0:
            raise ValueError(f"CTP's a must be at least 0 and d positive, got a = {self.a} and d = {self.d}.")
        self.a_j = self.b_j = ()
        self.n_constraints = 1

    def evaluate(self, X: Array, out: dict[str, Array]) -> None:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}).")
        f1 = X[:, 0]
        g = rastrigin_g(X)
        if self.variant == 1:
            f2 = g * np.exp(-f1 / g)
            boundaries = np.asarray(self.a_j)[None, :] * np.exp(-np.asarray(self.b_j)[None, :] * f1[:, None])
            C = f2[:, None] - boundaries
        else:
            # g (1 - f1 / g), without the rounding of the division.
            f2 = g - f1
            shifted = f2 - self.e
            sine, cosine = math.sin(self.theta), math.cos(self.theta)
            across = cosine * shifted - sine * f1
            along = sine * shifted + cosine * f1
            C = (across - self.a * np.abs(np.sin(self.b * np.pi * along**self.c)) ** self.d)[:, None]
        out["F"] = np.column_stack([f1, f2])
        out["G"] = -C


__all__ = ["CTP1_CONSTRAINTS", "PAPER_PARAMETERS", "CTPProblem", "ctp1_parameters", "rastrigin_g"]
