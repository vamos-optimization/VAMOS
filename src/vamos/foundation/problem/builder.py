"""
Friendly problem builder for VAMOS.

Provides ``make_problem()`` -- the simplest way to turn a plain Python
function into a fully compatible VAMOS problem, ready for ``optimize()``.

Example
-------
>>> from vamos import make_problem, optimize
>>> problem = make_problem(
...     lambda x: [x[0], 1 - x[0] ** 0.5],
...     n_var=2, n_obj=2,
...     bounds=[(0, 1), (0, 1)],
...     encoding="real",
... )
>>> result = optimize(problem, algorithm="nsgaii", max_evaluations=2000)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from vamos.foundation.encoding import EncodingLike, normalize_encoding
from vamos.foundation.exceptions import BoundsError, ConfigurationError, ProblemDimensionError
from vamos.foundation.problem.base import Problem, _coerce_decision_matrix, _output_matrix


class FunctionalProblem(Problem):
    """Problem wrapper that adapts a user function to the VAMOS ProblemProtocol.

    Created via :func:`make_problem` -- users should not instantiate this
    directly.
    """

    def __init__(
        self,
        fn: Callable[..., object],
        *,
        n_var: int,
        n_obj: int,
        xl: np.ndarray[Any, Any],
        xu: np.ndarray[Any, Any],
        encoding: EncodingLike,
        vectorized: bool,
        name: str,
        constraints_fn: Callable[..., object] | None,
        n_constraints: int,
    ) -> None:
        self.n_var = n_var
        self.n_obj = n_obj
        self.xl = xl
        self.xu = xu
        self.encoding = encoding
        self.name = name
        self._fn = fn
        self._vectorized = vectorized
        self._constraints_fn = constraints_fn
        self.n_constraints = n_constraints

    # ------------------------------------------------------------------
    # ProblemProtocol.evaluate
    # ------------------------------------------------------------------

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        """Evaluate the objective (and optional constraint) functions."""
        X = _coerce_decision_matrix(X, self.encoding)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}), got {X.shape}.")

        if self._vectorized:
            F_result = np.asarray(self._fn(X), dtype=float)
        else:
            # Elementwise adaptation: evaluate one solution at a time.
            results = [self._fn(X[i]) for i in range(X.shape[0])]
            F_result = np.asarray(results, dtype=float)

        N = X.shape[0]
        F_result = _output_matrix(F_result, N, self.n_obj, source="make_problem fn")

        # Write into pre-allocated buffer when available, else assign.
        F = out.get("F")
        if F is not None and F.shape == F_result.shape:
            F[:] = F_result
        else:
            out["F"] = F_result

        # ---- constraints ----
        if self._constraints_fn is not None:
            if self._vectorized:
                G_result = np.asarray(self._constraints_fn(X), dtype=float)
            else:
                g_results = [self._constraints_fn(X[i]) for i in range(X.shape[0])]
                G_result = np.asarray(g_results, dtype=float)

            G_result = _output_matrix(G_result, N, self.n_constraints, source="make_problem constraints fn")

            G = out.get("G")
            if G is not None and G.shape == G_result.shape:
                G[:] = G_result
            else:
                out["G"] = G_result

    def __repr__(self) -> str:
        return f"FunctionalProblem(name={self.name!r}, n_var={self.n_var}, n_obj={self.n_obj})"


# ======================================================================
# Public builder
# ======================================================================


def make_problem(
    fn: Callable[..., object],
    *,
    n_var: int,
    n_obj: int,
    bounds: Sequence[tuple[float, float]] | None = None,
    xl: float | Sequence[float] | np.ndarray[Any, Any] | None = None,
    xu: float | Sequence[float] | np.ndarray[Any, Any] | None = None,
    vectorized: bool = False,
    encoding: str,
    name: str | None = None,
    constraints: Callable[..., object] | None = None,
    n_constraints: int = 0,
) -> FunctionalProblem:
    """Create a VAMOS-compatible problem from a plain Python function.

    This is the friendliest way to define a custom optimization problem.
    Your function receives decision variables and returns objective values
    -- VAMOS handles bounds, protocol adaptation, and optional batched execution.

    Parameters
    ----------
    fn : callable
        Objective function.

        * **Scalar mode** (default, ``vectorized=False``): receives a 1-D
          array of shape ``(n_var,)`` and returns a list or array of
          ``n_obj`` objective values.
        * **Vectorized mode** (``vectorized=True``): receives a 2-D array
          of shape ``(N, n_var)`` and returns an array of shape
          ``(N, n_obj)``.

    n_var : int
        Number of decision variables.

    n_obj : int
        Number of objectives to minimize.

    bounds : sequence of (lower, upper) tuples, optional
        Per-variable bounds, e.g. ``[(0, 1), (0, 5)]``.  Must have
        length ``n_var``.  Mutually exclusive with *xl* / *xu*.

    xl, xu : float or array-like, optional
        Lower / upper bounds for all variables.  A scalar applies the
        same bound to every variable.  Mutually exclusive with *bounds*.

    vectorized : bool, default False
        If ``False`` VAMOS evaluates your scalar function one row at a time
        for compatibility. Set ``True`` only when your function already
        handles batches directly for real vectorized performance.

    encoding : str
        Variable encoding: ``"real"``, ``"binary"``, ``"integer"``,
        ``"permutation"``, or ``"mixed"``.

    name : str, optional
        Human-readable name shown in logs and results.  Defaults to the
        function name.

    constraints : callable, optional
        Constraint function following the same signature convention as
        *fn*.  Must return ``n_constraints`` values where
        ``g(x) <= 0`` is feasible.

    n_constraints : int, default 0
        Number of constraint values.  Required when *constraints* is
        provided.

    Returns
    -------
    FunctionalProblem
        A problem object ready to pass to ``vamos.optimize()``.

    Raises
    ------
    TypeError
        If *fn* is not callable.
    ValueError
        If *bounds* and *xl*/*xu* are both provided, if *bounds* length
        does not match *n_var*, or if *n_constraints* > 0 but no
        *constraints* callable is given.

    Examples
    --------
    Minimal two-objective problem::

        from vamos import make_problem, optimize

        problem = make_problem(
            lambda x: [x[0], 1 - x[0] ** 0.5],
            n_var=2, n_obj=2,
            bounds=[(0, 1), (0, 1)],
            encoding="real",
        )
        result = optimize(problem, algorithm="nsgaii", max_evaluations=2000)

    Vectorized for better performance::

        import numpy as np

        def my_objectives(X):
            f1 = X[:, 0]
            f2 = 1 - np.sqrt(X[:, 0])
            return np.column_stack([f1, f2])

        problem = make_problem(
            my_objectives,
            n_var=2, n_obj=2,
            bounds=[(0, 1), (0, 1)],
            vectorized=True,
            encoding="real",
        )

    With constraints (``g(x) <= 0`` is feasible)::

        problem = make_problem(
            lambda x: [x[0] + x[1], x[0] * x[1]],
            n_var=2, n_obj=2,
            bounds=[(0, 5), (0, 5)],
            encoding="real",
            constraints=lambda x: [x[0] + x[1] - 4],
            n_constraints=1,
        )
    """
    # ---- validate callable ----
    if not callable(fn):
        raise ConfigurationError(
            f"First argument must be a callable, got {type(fn).__name__}."
            "\n\nHint: make_problem(my_function, ...) where my_function(x) "
            "returns a list of objective values."
        )

    # ---- validate dimensions ----
    if not isinstance(n_var, int) or n_var < 1:
        raise ProblemDimensionError("n_var must be a positive integer.", n_var=n_var)
    if not isinstance(n_obj, int) or n_obj < 1:
        raise ProblemDimensionError("n_obj must be a positive integer.", n_obj=n_obj)

    # ---- resolve bounds ----
    if bounds is not None and (xl is not None or xu is not None):
        raise BoundsError("Use either 'bounds' or 'xl'/'xu', not both.\n\nHint: bounds=[(0, 1), (0, 1)] is equivalent to xl=0.0, xu=1.0")

    if bounds is not None:
        if len(bounds) != n_var:
            raise BoundsError(
                f"bounds has {len(bounds)} entries but n_var={n_var}.\n\nHint: provide exactly one (lower, upper) pair per variable."
            )
        for i, b in enumerate(bounds):
            if not (isinstance(b, (tuple, list)) and len(b) == 2):
                raise BoundsError(f"bounds[{i}] must be a (lower, upper) pair, got {b!r}.")
            if b[0] > b[1]:
                raise BoundsError(f"bounds[{i}]: lower bound ({b[0]}) > upper bound ({b[1]}).")
        xl_arr = np.array([b[0] for b in bounds], dtype=float)
        xu_arr = np.array([b[1] for b in bounds], dtype=float)
    else:
        if xl is None:
            xl = 0.0
        if xu is None:
            xu = 1.0
        try:
            xl_arr = np.broadcast_to(np.asarray(xl, dtype=float), (n_var,)).copy()
            xu_arr = np.broadcast_to(np.asarray(xu, dtype=float), (n_var,)).copy()
        except ValueError as exc:
            raise BoundsError(
                f"Could not broadcast xl/xu to shape ({n_var},).\n\nHint: pass scalars or one value per decision variable."
            ) from exc
        if np.any(xl_arr > xu_arr):
            raise BoundsError("Lower bounds must not exceed upper bounds.")

    # ---- validate constraints ----
    if constraints is not None:
        if not callable(constraints):
            raise ConfigurationError("'constraints' must be a callable.")
        if n_constraints < 1:
            raise ProblemDimensionError(
                "n_constraints must be >= 1 when a constraints function is provided."
                "\n\nHint: n_constraints is the number of constraint values "
                "your function returns."
            )
    if n_constraints > 0 and constraints is None:
        raise ProblemDimensionError(
            f"n_constraints={n_constraints} but no constraints function was provided."
            "\n\nHint: pass constraints=your_function where your_function(x) "
            "returns a list of n_constraints values (g(x) <= 0 is feasible)."
        )

    # ---- resolve name ----
    if name is None:
        name = getattr(fn, "__name__", None) or "custom_problem"
        if name == "<lambda>":
            name = "custom_problem"

    # ---- normalize encoding ----
    try:
        enc = normalize_encoding(encoding)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc

    return FunctionalProblem(
        fn,
        n_var=n_var,
        n_obj=n_obj,
        xl=xl_arr,
        xu=xu_arr,
        encoding=enc,
        vectorized=vectorized,
        name=name,
        constraints_fn=constraints,
        n_constraints=n_constraints,
    )


__all__ = ["make_problem", "FunctionalProblem"]
