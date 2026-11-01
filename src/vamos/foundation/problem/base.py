"""
Base class for class-based custom optimization problems.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vamos.foundation.encoding import EncodingLike, normalize_encoding
from vamos.foundation.exceptions import EvaluationError


def _coerce_decision_matrix(X: np.ndarray[Any, Any], encoding: EncodingLike) -> np.ndarray[Any, Any]:
    normalized = normalize_encoding(encoding)
    if normalized in {"binary", "integer", "permutation"}:
        return np.asarray(X)
    return np.asarray(X, dtype=float)


class Problem:
    """Base class for class-based custom optimization problems.

    Subclass this when your problem needs state — a dataset, distance matrix,
    simulator, or any data set up in ``__init__``.

    **Required:** set ``n_var``, ``n_obj``, ``xl``, ``xu`` in ``__init__``.
    **Optional:** override ``encoding`` and ``n_constraints`` as class-level
    attributes (not in ``__init__``).

    Example — unconstrained::

        import numpy as np
        from vamos import Problem, optimize

        class MyProblem(Problem):
            def __init__(self):
                self.n_var = 3
                self.n_obj = 2
                self.xl = np.zeros(3)
                self.xu = np.ones(3)

            def objectives(self, X: np.ndarray) -> np.ndarray:
                # X: (N, n_var) batch of candidate solutions
                f1 = np.sum(X ** 2, axis=1)
                f2 = np.sum((X - 1) ** 2, axis=1)
                return np.column_stack([f1, f2])

        result = optimize(MyProblem(), algorithm="nsgaii", max_evaluations=5000)

    Example — constrained::

        class MyConstrainedProblem(Problem):
            n_constraints = 1          # declare at class level

            def __init__(self):
                self.n_var = 3
                self.n_obj = 2
                self.xl = np.zeros(3)
                self.xu = np.ones(3)

            def objectives(self, X):
                f1 = np.sum(X ** 2, axis=1)
                f2 = np.sum((X - 1) ** 2, axis=1)
                return np.column_stack([f1, f2])

            def constraints(self, X):
                # Sign convention: g(x) <= 0 means feasible.
                g = np.sum(X, axis=1) - 2.0   # sum(x) <= 2
                return g.reshape(-1, 1)
    """

    # ------------------------------------------------------------------
    # Required instance attributes — must be set in subclass __init__
    # ------------------------------------------------------------------

    n_var: int
    n_obj: int
    xl: float | int | np.ndarray[Any, Any]
    xu: float | int | np.ndarray[Any, Any]

    # ------------------------------------------------------------------
    # Class-level defaults — override at class body level, not in __init__
    # ------------------------------------------------------------------

    encoding: EncodingLike = "real"
    """Variable encoding.  Supported values: ``"real"``, ``"integer"``,
    ``"binary"``, ``"permutation"``, ``"mixed"``.  Default: ``"real"``."""

    n_constraints: int = 0
    """Number of inequality constraints.  Default: ``0`` (unconstrained)."""

    # ------------------------------------------------------------------
    # User-overridable interface
    # ------------------------------------------------------------------

    def objectives(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Compute objective values for a batch of solutions.

        Override this method in your subclass.

        Parameters
        ----------
        X : np.ndarray
            Decision matrix of shape ``(N, n_var)``.

        Returns
        -------
        np.ndarray
            Objective array of shape ``(N, n_obj)`` to minimize. A
            single-objective problem may return a 1-D array of length ``N``.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement objectives(self, X).")

    def constraints(self, X: np.ndarray[Any, Any]) -> np.ndarray[Any, Any] | None:
        """Compute constraint violations for a batch of solutions.

        Override this method when your problem has inequality constraints.

        Parameters
        ----------
        X : np.ndarray
            Decision matrix of shape ``(N, n_var)``.

        Returns
        -------
        np.ndarray | None
            Constraint array of shape ``(N, n_constraints)`` where negative
            values indicate feasibility, or ``None`` for unconstrained problems.
        """
        return None

    # ------------------------------------------------------------------
    # Framework entry point — do not override
    # ------------------------------------------------------------------

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        """Framework evaluation entry point.  Override :meth:`objectives`
        (and optionally :meth:`constraints`) instead of this method."""
        X = _coerce_decision_matrix(X, self.encoding)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}), got {X.shape}.")

        # --- objectives ---
        N = X.shape[0]
        F_computed = np.asarray(self.objectives(X), dtype=float)
        if F_computed.ndim == 0:
            F_computed = F_computed.reshape(1, 1)
        elif F_computed.ndim == 1:
            F_computed = F_computed.reshape(-1, self.n_obj)
        if F_computed.shape != (N, self.n_obj):
            raise ValueError(f"{type(self).__name__}.objectives() returned shape {F_computed.shape}, expected ({N}, {self.n_obj}).")
        F_buf = out.get("F")
        if F_buf is not None and F_buf.shape == F_computed.shape:
            F_buf[:] = F_computed
        else:
            out["F"] = F_computed

        # --- constraints ---
        if self.n_constraints <= 0 and type(self).constraints is not Problem.constraints:
            # An overridden constraints() without n_constraints would be ignored: the problem would run unconstrained.
            G_undeclared = self.constraints(X)
            if G_undeclared is not None and np.size(G_undeclared) > 0:
                raise EvaluationError(
                    f"{type(self).__name__}.constraints() returned values but n_constraints=0, so VAMOS would optimize "
                    "it as unconstrained. Set the class attribute n_constraints to the number of constraints."
                )
        if self.n_constraints > 0:
            G_computed = self.constraints(X)
            if G_computed is None:
                raise ValueError(
                    f"{type(self).__name__}.constraints() returned None but "
                    f"n_constraints={self.n_constraints}. Override constraints() "
                    "to return an array of shape (N, n_constraints)."
                )
            G_computed = np.asarray(G_computed, dtype=float)
            if G_computed.ndim == 0:
                G_computed = G_computed.reshape(1, 1)
            elif G_computed.ndim == 1:
                G_computed = G_computed.reshape(-1, self.n_constraints)
            if G_computed.shape != (N, self.n_constraints):
                raise ValueError(
                    f"{type(self).__name__}.constraints() returned shape {G_computed.shape}, expected ({N}, {self.n_constraints})."
                )
            G_buf = out.get("G")
            if G_buf is not None and G_buf.shape == G_computed.shape:
                G_buf[:] = G_computed
            else:
                out["G"] = G_computed


__all__ = ["Problem"]
