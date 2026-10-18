"""SMPSO operator building.

This module provides factory functions for building SMPSO operators:
- Mutation operator (polynomial mutation / turbulence)
- Repair operator for bounds handling
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from vamos.engine.algorithm.smpso.helpers import resolve_repair
from vamos.engine.operators.impl.mixed import mixed_mutation
from vamos.engine.operators.impl.real import ClampRepair, PolynomialMutation, VariationWorkspace
from vamos.engine.variation import prepare_mutation_params
from vamos.foundation.encoding import normalize_encoding

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import VariationKernels
    from vamos.foundation.problem.types import ProblemProtocol


__all__ = [
    "build_mutation_operator",
    "build_repair_operator",
]


def build_mutation_operator(
    config: dict[str, Any],
    encoding: str,
    n_var: int,
    xl: NDArray[np.float64],
    xu: NDArray[np.float64],
    problem: ProblemProtocol | None = None,
    *,
    variation_kernels: VariationKernels | None = None,
) -> Any:
    """Build the mutation (turbulence) operator for SMPSO.

    SMPSO uses polynomial mutation to add turbulence to particle positions,
    helping to escape local optima.

    Parameters
    ----------
    config : dict
        Algorithm configuration with 'mutation' key.
    encoding : str
        Variable encoding ("real" or "mixed").
    n_var : int
        Number of decision variables.
    xl : NDArray[np.generic]
        Lower bounds.
    xu : NDArray[np.generic]
        Upper bounds.
    variation_kernels : VariationKernels | None
        Optional backend kernels for the polynomial mutation arithmetic.

    Returns
    -------
    Any
        Configured mutation operator.

    Raises
    ------
    ValueError
        If mutation type is not supported.
    """
    normalized = normalize_encoding(encoding)
    mut_method, mut_params = config.get("mutation", ("polynomial", {}))
    mut_method = str(mut_method).lower()
    if mut_method == "pm":
        mut_method = "polynomial"

    if normalized == "mixed":
        if mut_method not in {"mixed", "gaussian"}:
            raise ValueError(f"Unsupported SMPSO mutation '{mut_method}' for mixed encoding. Use 'mixed' or 'gaussian'.")
        mixed_spec = getattr(problem, "mixed_spec", None) if problem is not None else None
        if mixed_spec is None:
            raise ValueError("SMPSO mixed encoding requires problem.mixed_spec.")
        mut_params = prepare_mutation_params(dict(mut_params or {}), normalized, n_var)
        prob_raw = mut_params.get("prob")
        prob = float(prob_raw) if prob_raw is not None else 1.0 / max(1, n_var)
        spec = dict(mixed_spec)
        for key, value in mut_params.items():
            if key == "prob":
                continue
            spec[key] = value

        class _MixedMutationOperator:
            def __init__(self, mutation_prob: float, spec: dict[str, NDArray[np.generic]]) -> None:
                self._prob = float(mutation_prob)
                self._spec = spec

            def __call__(self, X: NDArray[np.generic], rng: np.random.Generator) -> NDArray[np.generic]:
                mixed_mutation(X, self._prob, self._spec, rng)
                return X

        return _MixedMutationOperator(prob, spec)

    if mut_method != "polynomial":
        raise ValueError(f"Unsupported SMPSO mutation '{mut_method}'.")

    mut_params = prepare_mutation_params(dict(mut_params or {}), normalized, n_var)
    workspace = VariationWorkspace()
    prob_raw = mut_params.get("prob")
    prob = float(prob_raw) if prob_raw is not None else 1.0 / max(1, n_var)
    eta_raw = mut_params.get("eta")
    eta = float(eta_raw) if eta_raw is not None else 20.0

    return PolynomialMutation(
        prob_mutation=prob,
        eta=eta,
        lower=xl,
        upper=xu,
        workspace=workspace,
        variation_kernels=variation_kernels,
    )


def build_repair_operator(config: dict[str, Any], encoding: str) -> Any | None:
    """Build the repair operator from configuration.

    Parameters
    ----------
    config : dict
        Algorithm configuration with optional 'repair' key.

    Returns
    -------
    Any or None
        Repair operator instance or None if not configured.
    """
    normalized = normalize_encoding(encoding)
    repair_cfg = config.get("repair", "auto")
    if repair_cfg == "auto":
        if normalized == "real":
            return ClampRepair()
        return None
    return resolve_repair(repair_cfg)
