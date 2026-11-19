"""MODAct: Picard and Schiffmann's constrained real-world problems from the design of actuators.

C. Picard and J. Schiffmann, "Realistic constrained multiobjective optimization benchmark problems from design",
IEEE Transactions on Evolutionary Computation 25(2), 234-246 (2021), https://doi.org/10.1109/TEVC.2020.3020046.

The 20 problems design an electro-mechanical actuator, a stepper motor and three spur-gear stages, with 20
continuous variables; the integer choices (the motor and each gear's number of teeth) are the integer parts of
variables whose fractional parts set a continuous one. A problem's name joins its objectives, C (cost), T (torque
excess), S (safety factor), E (efficiency) and I (transmission ratio), to its constraint level, 1 to 4, which has
7, 8, 10 or 9 constraints (the paper's Table III).

VAMOS does not implement the problems: every evaluation calls the authors' ``modact`` package
(https://github.com/epfl-lamd/modact, GPL-3.0), which VAMOS neither includes nor redistributes. A problem imports
it when it is created, and raises ``ImportError`` with the command that installs it when it is missing:
``python -m pip install "vamos-optimization[modact]"``, or ``python -m pip install modact``.

The package's ``weights`` and ``c_weights`` give VAMOS's conventions:

- ``weights[j]`` is -1 for an objective that MODAct minimizes (C and I) and 1 for one that it maximizes (T, S and
  E). ``F[:, j]`` is ``-weights[j] * f_j``: the maximized objectives change sign, and every column is minimized.
- ``c_weights[k]`` is -1 for a constraint ``g_k >= 0`` and 1 for a constraint ``g_k <= 0``. ``G[:, k]`` is
  ``c_weights[k] * g_k``, so ``G <= 0`` is feasible.

Both only multiply by 1 or -1, so ``F`` and ``G`` hold the package's values, sign aside; pymoo's
``pymoo.problems.multi.modact`` converts them the same way.
"""

from __future__ import annotations

import importlib
from importlib import metadata
from types import ModuleType
from typing import Any

import numpy as np

from vamos.foundation.problem.base import Problem

Array = np.ndarray[Any, Any]

N_VAR = 20
# The objectives of each problem class, which MODAct minimizes (cost, transmission ratio) or maximizes.
OBJECTIVES: dict[str, tuple[str, ...]] = {
    "cs": ("cost", "safety factor"),
    "ct": ("cost", "torque excess"),
    "cts": ("cost", "torque excess", "safety factor"),
    "ctse": ("cost", "torque excess", "safety factor", "efficiency"),
    "ctsei": ("cost", "torque excess", "safety factor", "efficiency", "transmission ratio"),
}
# The objectives that MODAct maximizes, and VAMOS minimizes with the opposite sign.
MAXIMIZED: tuple[str, ...] = ("torque excess", "safety factor", "efficiency")
# The number of constraints of each constraint level (the paper's Table III).
CONSTRAINT_LEVELS: dict[int, int] = {1: 7, 2: 8, 3: 10, 4: 9}
# MODAct's name of each problem: (number of objectives, number of constraints).
MODACT: dict[str, tuple[int, int]] = {
    "cs1": (2, 7),
    "cs2": (2, 8),
    "cs3": (2, 10),
    "cs4": (2, 9),
    "ct1": (2, 7),
    "ct2": (2, 8),
    "ct3": (2, 10),
    "ct4": (2, 9),
    "cts1": (3, 7),
    "cts2": (3, 8),
    "cts3": (3, 10),
    "cts4": (3, 9),
    "ctse1": (4, 7),
    "ctse2": (4, 8),
    "ctse3": (4, 10),
    "ctse4": (4, 9),
    "ctsei1": (5, 7),
    "ctsei2": (5, 8),
    "ctsei3": (5, 10),
    "ctsei4": (5, 9),
}
INSTALL_COMMAND = 'python -m pip install "vamos-optimization[modact]"'


def load_modact() -> ModuleType:
    """The ``modact.problems`` module of the installed ``modact`` package.

    Raises ``ImportError`` with the command that installs the package when it, or one of its dependencies, cannot
    be imported.
    """
    try:
        return importlib.import_module("modact.problems")
    except ImportError as exc:
        raise ImportError(
            "The MODAct problems are evaluated by the separately installed modact package (GPL-3.0), which VAMOS "
            f"does not include, and it cannot be imported ({exc}). Install it with: {INSTALL_COMMAND} "
            "(or python -m pip install modact)."
        ) from exc


def _signs(values: object, name: str, key: str) -> Array:
    signs = np.asarray(values, dtype=float)
    if not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError(f"modact's {name} of {key} must be -1 or 1; got {signs.tolist()}.")
    return signs


class MODActProblem(Problem):
    """The MODAct problem ``name``, ``"cs1"`` to ``"ctsei4"``, evaluated by the installed ``modact`` package.

    ``objective_signs`` (``-weights``) and ``constraint_signs`` (``c_weights``) are the factors that turn the
    package's objectives and constraints into ``F`` and ``G``. ``objective_signs`` also converts a front in
    MODAct's signs, such as the authors' best-known fronts on Zenodo, to VAMOS's. ``modact_version`` is the
    version of the installed package, which a run's ``environment.json`` records too.
    """

    def __init__(self, name: str) -> None:
        modact_name = str(name).lower()
        if modact_name not in MODACT:
            raise ValueError(f"Unknown MODAct problem {name!r}; the problems are: {', '.join(MODACT)}.")
        n_obj, n_constraints = MODACT[modact_name]
        problem = load_modact().get_problem(modact_name)
        lower, upper = (np.asarray(bound, dtype=float) for bound in problem.bounds())
        objective_signs = -_signs(problem.weights, "weights", modact_name)
        constraint_signs = _signs(problem.c_weights, "c_weights", modact_name)
        found = (lower.size, upper.size, objective_signs.size, constraint_signs.size)
        if found != (N_VAR, N_VAR, n_obj, n_constraints):
            raise ValueError(
                f"The installed modact defines {modact_name} with {found[0]} variables, {found[2]} objectives and "
                f"{found[3]} constraints; VAMOS registers it with {N_VAR}, {n_obj} and {n_constraints}, as the paper "
                "and modact 1.0.1 define it."
            )
        self.key = f"modact_{modact_name}"
        self.modact_name = modact_name
        self.n_var = N_VAR
        self.n_obj = n_obj
        self.n_constraints = n_constraints
        self.xl = lower
        self.xu = upper
        self.objective_signs = objective_signs
        self.constraint_signs = constraint_signs
        self.modact_version = _installed_version()
        self._problem = problem

    def evaluate(self, X: Array, out: dict[str, Array]) -> None:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_var:
            raise ValueError(f"Expected decision matrix of shape (N, {self.n_var}).")
        F = np.empty((X.shape[0], self.n_obj))
        G = np.empty((X.shape[0], self.n_constraints))
        for row, x in enumerate(X):
            # modact evaluates one design at a time; a copy keeps the caller's population out of its reach.
            f, g = self._problem(x.copy())
            F[row] = f
            G[row] = g
        out["F"] = F * self.objective_signs
        out["G"] = G * self.constraint_signs


def _installed_version() -> str | None:
    try:
        return metadata.version("modact")
    except metadata.PackageNotFoundError:
        return None


__all__ = ["CONSTRAINT_LEVELS", "INSTALL_COMMAND", "MAXIMIZED", "MODACT", "MODActProblem", "N_VAR", "OBJECTIVES", "load_modact"]
