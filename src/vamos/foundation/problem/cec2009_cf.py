"""CEC2009 CF2-CF10, the constrained problems of the CEC 2009 competition.

Q. Zhang, A. Zhou, S. Zhao, P. N. Suganthan, W. Liu and S. Tiwari, "Multiobjective optimization test
instances for the CEC 2009 special session and competition", Technical Report CES-487, University of
Essex and Nanyang Technological University, 2008 (revised on April 20, 2009). They follow jMetal's
implementation (``org.uma.jmetal.problem.multiobjective.cec2009``), which follows the competition's C
code. The code states each constraint as ``c(x) >= 0``; ``out["G"]`` holds ``-c(x)`` so that, as
everywhere in VAMOS, ``G <= 0`` means feasible. CF1 is in :mod:`vamos.foundation.problem.cec2009`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from vamos.foundation.problem.cec2009 import _CEC2009Base


def _sigmoid(t: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """The constraint of CF2 and CF4 from its argument: ``t / (1 + exp(4 |t|))``, with the sign of ``t``."""
    return np.asarray(t / (1.0 + np.exp(4.0 * np.abs(t))))


def _h2(t: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """The term of x2 in CF4 and CF5: ``|t|`` below ``1.5 - 0.75 sqrt(2)``, ``0.125 + (t - 1)^2`` from there on."""
    threshold = 1.5 - 0.75 * math.sqrt(2.0)
    return np.where(t < threshold, np.abs(t), 0.125 + (t - 1.0) * (t - 1.0))


def _signed_sqrt(value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """``sign(value) sqrt(|value|)``, the bound of y2 and y4 in the constraints of CF6 and CF7."""
    root = np.sqrt(np.abs(value))
    return np.where(value > 0.0, root, -root)


def _multimodal(y: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """``2 y^2 - cos(4 pi y) + 1``, the distance term of CF5 and CF7."""
    return np.asarray(2.0 * y * y - np.cos(4.0 * np.pi * y) + 1.0)


class _CEC2009CFBase(_CEC2009Base):
    """The set-up of CF2-CF10: the position variables in ``[0, 1]`` and the others in ``[lower, upper]``."""

    def __init__(self, n_var: int, *, n_obj: int, n_var_min: int, lower: float, upper: float) -> None:
        super().__init__(n_var=n_var, n_obj=n_obj, n_var_min=n_var_min, lower_rest=lower, upper_rest=upper)

    def _angles(self, x0: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """``6 pi x1 + j pi / n`` for j = 2, ..., n."""
        j = np.arange(2, self.n_var + 1)
        return np.asarray(6.0 * np.pi * x0[:, None] + j * np.pi / self.n_var)

    def _shifted_sine(self, x0: np.ndarray[Any, Any], j: int) -> np.ndarray[Any, Any]:
        """``sin(6 x1 pi + j pi / n)``, as the constraints of CF4-CF7 write it."""
        return np.asarray(np.sin(6.0 * x0 * np.pi + j * np.pi / self.n_var))


class CEC2009_CF2(_CEC2009CFBase):
    """Constrained CF2 from the CEC2009 competition, on ``[0, 1] x [-1, 1]^(n-1)``.

    Its front is the point (0, 1) and the parts of ``f2 = 1 - sqrt(f1)`` with f1 in [1/16, 1/4] and [9/16, 1].
    """

    n_constraints = 1

    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var, n_obj=2, n_var_min=3, lower=-1.0, upper=1.0)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X, F = self._prepare(X, out)
        x0, rest = X[:, 0], X[:, 1:]
        _, even_mask, odd_mask = self._split_even_odd()
        angles = self._angles(x0)
        y_odd = rest[:, odd_mask] - np.sin(angles[:, odd_mask])
        y_even = rest[:, even_mask] - np.cos(angles[:, even_mask])
        f0 = x0 + 2.0 * np.sum(y_odd * y_odd, axis=1) / int(odd_mask.sum())
        f1 = 1.0 - np.sqrt(x0) + 2.0 * np.sum(y_even * y_even, axis=1) / int(even_mask.sum())
        t = f1 + np.sqrt(f0) - np.sin(2.0 * np.pi * (np.sqrt(f0) - f1 + 1.0)) - 1.0
        F[:, 0] = f0
        F[:, 1] = f1
        out["G"] = -_sigmoid(t)[:, None]


class CEC2009_CF3(_CEC2009CFBase):
    """Constrained CF3 from the CEC2009 competition, on ``[0, 1] x [-2, 2]^(n-1)``.

    Its front is the point (0, 1) and the parts of ``f2 = 1 - f1^2`` with f1 in [1/2, sqrt(1/2)] and
    [sqrt(3/4), 1].
    """

    n_constraints = 1

    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var, n_obj=2, n_var_min=3, lower=-2.0, upper=2.0)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X, F = self._prepare(X, out)
        x0, rest = X[:, 0], X[:, 1:]
        j, even_mask, odd_mask = self._split_even_odd()
        y = rest - np.sin(self._angles(x0))
        p = np.cos(20.0 * y * np.pi / np.sqrt(j))
        y2 = y * y
        count_odd, count_even = int(odd_mask.sum()), int(even_mask.sum())
        f0 = x0 + 2.0 * (4.0 * np.sum(y2[:, odd_mask], axis=1) - 2.0 * np.prod(p[:, odd_mask], axis=1) + 2.0) / count_odd
        f1 = 1.0 - x0 * x0 + 2.0 * (4.0 * np.sum(y2[:, even_mask], axis=1) - 2.0 * np.prod(p[:, even_mask], axis=1) + 2.0) / count_even
        F[:, 0] = f0
        F[:, 1] = f1
        out["G"] = -(f1 + f0 * f0 - np.sin(2.0 * np.pi * (f0 * f0 - f1 + 1.0)) - 1.0)[:, None]


class CEC2009_CF4(_CEC2009CFBase):
    """Constrained CF4 from the CEC2009 competition, on ``[0, 1] x [-2, 2]^(n-1)``.

    The objectives add the distance terms instead of averaging them, and the constraint is
    ``y2 >= 0.5 x1 - 0.25``. Its front is ``f2 = 1 - f1`` on [0, 0.5], ``0.75 - 0.5 f1`` on (0.5, 0.75] and
    ``1.125 - f1`` on (0.75, 1].
    """

    n_constraints = 1

    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var, n_obj=2, n_var_min=2, lower=-2.0, upper=2.0)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X, F = self._prepare(X, out)
        x0, rest = X[:, 0], X[:, 1:]
        j, even_mask, odd_mask = self._split_even_odd()
        y = rest - np.sin(self._angles(x0))
        terms = np.where(j == 2, _h2(y), y * y)
        t = X[:, 1] - self._shifted_sine(x0, 2) - 0.5 * x0 + 0.25
        F[:, 0] = x0 + np.sum(terms[:, odd_mask], axis=1)
        F[:, 1] = 1.0 - x0 + np.sum(terms[:, even_mask], axis=1)
        out["G"] = -_sigmoid(t)[:, None]


class CEC2009_CF5(_CEC2009CFBase):
    """Constrained CF5 from the CEC2009 competition, on ``[0, 1] x [-2, 2]^(n-1)``.

    The constraint is ``y2 >= 0.5 x1 - 0.25`` and the front is that of CF4.
    """

    n_constraints = 1

    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var, n_obj=2, n_var_min=2, lower=-2.0, upper=2.0)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X, F = self._prepare(X, out)
        x0, rest = X[:, 0], X[:, 1:]
        j, even_mask, odd_mask = self._split_even_odd()
        angles = self._angles(x0)
        scale = 0.8 * x0[:, None]
        y = rest - scale * np.where(odd_mask, np.cos(angles), np.sin(angles))
        terms = np.where(j == 2, _h2(y), _multimodal(y))
        F[:, 0] = x0 + np.sum(terms[:, odd_mask], axis=1)
        F[:, 1] = 1.0 - x0 + np.sum(terms[:, even_mask], axis=1)
        out["G"] = -(X[:, 1] - 0.8 * x0 * self._shifted_sine(x0, 2) - 0.5 * x0 + 0.25)[:, None]


class _CEC2009CF67(_CEC2009CFBase):
    """CF6 and CF7: the constraints ``y2 >= sign(g1) sqrt(|g1|)`` and ``y4 >= sign(g2) sqrt(|g2|)``."""

    n_constraints = 2

    def __init__(self, n_var: int) -> None:
        super().__init__(n_var, n_obj=2, n_var_min=4, lower=-2.0, upper=2.0)

    def _constraints(self, X: np.ndarray[Any, Any], scale: np.ndarray[Any, Any] | float) -> np.ndarray[Any, Any]:
        x0 = X[:, 0]
        g1 = (x0 - 0.5) * (1.0 - x0)
        g2 = 0.25 * np.sqrt(1.0 - x0) - 0.5 * (1.0 - x0)
        c1 = X[:, 1] - scale * self._shifted_sine(x0, 2) - _signed_sqrt(g1)
        c2 = X[:, 3] - scale * self._shifted_sine(x0, 4) - _signed_sqrt(g2)
        return -np.column_stack([c1, c2])


class CEC2009_CF6(_CEC2009CF67):
    """Constrained CF6 from the CEC2009 competition, on ``[0, 1] x [-2, 2]^(n-1)``.

    Its front is ``f2 = (1 - f1)^2`` on [0, 0.5], ``0.5 (1 - f1)`` on (0.5, 0.75] and ``0.25 sqrt(1 - f1)`` on
    (0.75, 1].
    """

    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X, F = self._prepare(X, out)
        x0, rest = X[:, 0], X[:, 1:]
        _, even_mask, odd_mask = self._split_even_odd()
        angles = self._angles(x0)
        y = rest - 0.8 * x0[:, None] * np.where(odd_mask, np.cos(angles), np.sin(angles))
        y2 = y * y
        F[:, 0] = x0 + np.sum(y2[:, odd_mask], axis=1)
        F[:, 1] = (1.0 - x0) * (1.0 - x0) + np.sum(y2[:, even_mask], axis=1)
        out["G"] = self._constraints(X, 0.8 * x0)


class CEC2009_CF7(_CEC2009CF67):
    """Constrained CF7 from the CEC2009 competition, on ``[0, 1] x [-2, 2]^(n-1)``.

    The distance terms of x2 and x4 are ``y^2``, the others ``2 y^2 - cos(4 pi y) + 1``; the front is that of
    CF6.
    """

    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X, F = self._prepare(X, out)
        x0, rest = X[:, 0], X[:, 1:]
        j, even_mask, odd_mask = self._split_even_odd()
        angles = self._angles(x0)
        y = rest - np.where(odd_mask, np.cos(angles), np.sin(angles))
        terms = np.where((j == 2) | (j == 4), y * y, _multimodal(y))
        F[:, 0] = x0 + np.sum(terms[:, odd_mask], axis=1)
        F[:, 1] = (1.0 - x0) * (1.0 - x0) + np.sum(terms[:, even_mask], axis=1)
        out["G"] = self._constraints(X, 1.0)


class _CEC2009CFSphere(_CEC2009CFBase):
    """CF8-CF10: three objectives on the unit sphere and the constraint

    ``(f1^2 + f2^2) / (1 - f3^2) - a s(N pi ((f1^2 - f2^2) / (1 - f3^2) + 1)) - 1 >= 0``, ``N = 2``,

    with ``s = |sin|`` (CF8) or ``sin``. Where ``f3 = 1`` the competition's code divides by zero; as in jMetal,
    the constraint is then 0 (satisfied), which happens at the top (0, 0, 1) of the front and, by rounding,
    next to it.
    """

    n_constraints = 1
    _A = 1.0
    _ABSOLUTE_SINE = False
    _MULTIMODAL = False

    def __init__(self, n_var: int, *, bound: float) -> None:
        super().__init__(n_var, n_obj=3, n_var_min=5, lower=-bound, upper=bound)

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        X, F = self._prepare(X, out)
        x0, x1, rest = X[:, 0], X[:, 1], X[:, 2:]
        j, mod1, mod2, mod0 = self._split_mod3()
        y = rest - 2.0 * x1[:, None] * np.sin(2.0 * np.pi * x0[:, None] + j * np.pi / self.n_var)
        h = 4.0 * y * y - np.cos(8.0 * np.pi * y) + 1.0 if self._MULTIMODAL else y * y
        cosine = np.cos(0.5 * np.pi * x0)
        F[:, 0] = cosine * np.cos(0.5 * np.pi * x1) + 2.0 * np.sum(h[:, mod1], axis=1) / int(mod1.sum())
        F[:, 1] = cosine * np.sin(0.5 * np.pi * x1) + 2.0 * np.sum(h[:, mod2], axis=1) / int(mod2.sum())
        F[:, 2] = np.sin(0.5 * np.pi * x0) + 2.0 * np.sum(h[:, mod0], axis=1) / int(mod0.sum())

        f1, f2, f3 = F[:, 0], F[:, 1], F[:, 2]
        denominator = 1.0 - f3 * f3
        with np.errstate(divide="ignore", invalid="ignore"):
            sine = np.sin(2.0 * np.pi * ((f1 * f1 - f2 * f2) / denominator + 1.0))
            if self._ABSOLUTE_SINE:
                sine = np.abs(sine)
            constraint = (f1 * f1 + f2 * f2) / denominator - self._A * sine - 1.0
        out["G"] = -np.where(denominator == 0.0, 0.0, constraint)[:, None]


class CEC2009_CF8(_CEC2009CFSphere):
    """Constrained CF8 from the CEC2009 competition, on ``[0, 1]^2 x [-4, 4]^(n-2)``, with ``a = 4`` and ``|sin|``.

    Its front is the five curves of the unit sphere with x2 in {0, 1/3, 1/2, 2/3, 1}.
    """

    _A = 4.0
    _ABSOLUTE_SINE = True

    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var, bound=4.0)


class CEC2009_CF9(_CEC2009CFSphere):
    """Constrained CF9 from the CEC2009 competition, on ``[0, 1]^2 x [-2, 2]^(n-2)``, with ``a = 3``.

    Its front is the parts of the unit sphere with x2 in [0, 1/3], [1/2, 2/3] or {1}.
    """

    _A = 3.0

    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var, bound=2.0)


class CEC2009_CF10(_CEC2009CFSphere):
    """Constrained CF10 from the CEC2009 competition, on ``[0, 1]^2 x [-2, 2]^(n-2)``, with ``a = 1``.

    The distance terms are ``4 y^2 - cos(8 pi y) + 1``, and the front is that of CF9.
    """

    _MULTIMODAL = True

    def __init__(self, n_var: int = 10) -> None:
        super().__init__(n_var, bound=2.0)


__all__ = [
    "CEC2009_CF2",
    "CEC2009_CF3",
    "CEC2009_CF4",
    "CEC2009_CF5",
    "CEC2009_CF6",
    "CEC2009_CF7",
    "CEC2009_CF8",
    "CEC2009_CF9",
    "CEC2009_CF10",
]
