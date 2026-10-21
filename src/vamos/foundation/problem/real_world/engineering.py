from __future__ import annotations

from typing import Any

import numpy as np


class WeldedBeamDesignProblem:
    """
    Mixed-encoding welded-beam style engineering design problem.

    Variables (n_var = 6):
        x0: weld thickness h in [0.125, 5] (real)
        x1: length l in [0.1, 10] (real)
        x2: flange thickness t in [0.1, 10] (real)
        x3: width b in [0.125, 5] (real)
        x4: material choice {0,1,2} (categorical, rounded)
        x5: stiffener count in [2, 6] (integer, rounded)

    Objectives (to minimize):
        f1: fabrication cost (material-dependent)
        f2: deflection proxy (lower is better)

    Constraints (stored in out['G'], <= 0 feasible):
        g1: shear stress - limit
        g2: normal stress - limit
        g3: deflection - limit
        g4: geometric coupling (h <= b)
    """

    n_constraints = 4

    def __init__(self) -> None:
        self.n_var = 6
        self.n_obj = 2
        self.encoding = "mixed"
        self.xl = np.array([0.125, 0.1, 0.1, 0.125, 0.0, 2.0], dtype=float)
        self.xu = np.array([5.0, 10.0, 10.0, 5.0, 2.0, 6.0], dtype=float)
        self._material_factor = np.array([1.0, 1.15, 1.3])
        self.mixed_spec: dict[str, np.ndarray[Any, Any]] = {
            "real_idx": np.array([0, 1, 2, 3], dtype=int),
            "int_idx": np.array([5], dtype=int),
            "cat_idx": np.array([4], dtype=int),
            "real_lower": np.array([0.125, 0.1, 0.1, 0.125], dtype=float),
            "real_upper": np.array([5.0, 10.0, 10.0, 5.0], dtype=float),
            "int_lower": np.array([2], dtype=int),
            "int_upper": np.array([6], dtype=int),
            "cat_cardinality": np.array([3], dtype=int),
        }

    def evaluate(self, X: np.ndarray[Any, Any], out: dict[str, np.ndarray[Any, Any]]) -> None:
        h = X[:, 0]
        l = X[:, 1]
        t = X[:, 2]
        b = X[:, 3]
        material_idx = np.clip(np.rint(X[:, 4]), 0, self._material_factor.size - 1).astype(int)
        stiffeners = np.clip(np.rint(X[:, 5]), 2, 6)

        # Core welded beam objectives
        cost_base = 1.10471 * h * h + 0.04811 * t * b * (14.0 + l)
        cost = cost_base * self._material_factor[material_idx] + 5.0 * stiffeners
        delta = 2.1952 / (np.maximum(t, 1e-6) ** 3 * np.maximum(b, 1e-6) * stiffeners)

        # Stress calculations (standard welded-beam surrogate)
        tau_prime = 6000.0 / (np.sqrt(2) * h * l)
        M = 6000.0 * (14.0 + 0.5 * l)
        R = np.sqrt(0.25 * l * l + ((h + t) / 2.0) ** 2)
        J = 2.0 * (np.sqrt(2) * h * l) * (0.25 * l * l + R * R)
        tau = np.sqrt(tau_prime**2 + (M * R / J) ** 2 + 2 * tau_prime * M * R / J)
        sigma = 504000.0 / (t * b * b)

        g1 = tau - 13600.0
        g2 = sigma - 30000.0
        g3 = delta - 0.25
        g4 = h - b  # enforce h <= b

        F = out["F"]
        F[:, 0] = cost
        F[:, 1] = delta
        out["G"] = np.vstack([g1, g2, g3, g4]).T


__all__ = ["WeldedBeamDesignProblem"]
