# operators/policies/ibea.py
"""
Operator building for IBEA.

This module handles the construction of variation pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from vamos.engine.operators.impl.real import VariationWorkspace
from vamos.engine.variation import VariationPipeline, prepare_mutation_params
from vamos.engine.variation.protocol import RepairConfigValue
from vamos.foundation.encoding import normalize_encoding

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import VariationKernels
    from vamos.foundation.problem.types import ProblemProtocol


def build_variation_pipeline(
    cfg: dict[str, Any],
    encoding: str,
    n_var: int,
    xl: NDArray[np.generic],
    xu: NDArray[np.generic],
    problem: ProblemProtocol,
    *,
    variation_kernels: VariationKernels | None = None,
) -> VariationPipeline:
    """Build variation pipeline for IBEA.

    Parameters
    ----------
    cfg : dict[str, Any]
        Algorithm configuration with 'crossover' and 'mutation' keys.
    encoding : str
        Problem encoding type.
    n_var : int
        Number of decision variables.
    xl : NDArray[np.generic]
        Lower bounds.
    xu : NDArray[np.generic]
        Upper bounds.
    problem : ProblemProtocol
        The optimization problem.
    variation_kernels : VariationKernels | None
        Optional backend kernels for SBX and polynomial mutation arithmetic.

    Returns
    -------
    VariationPipeline
        Configured variation pipeline.
    """
    cross_method, cross_params = cfg["crossover"]
    cross_method = cross_method.lower()
    cross_params = dict(cross_params)

    mut_method, mut_params = cfg["mutation"]
    mut_method = mut_method.lower()
    normalized = normalize_encoding(encoding)
    mut_params = prepare_mutation_params(
        cast(dict[str, float | int | str | None], dict(mut_params)),
        normalized,
        n_var,
        prob_factor=cast(float | None, cfg.get("mutation_prob_factor")),
    )

    variation_workspace = VariationWorkspace()
    variation = VariationPipeline(
        encoding=normalized,
        cross_method=cross_method,
        cross_params=cross_params,
        mut_method=mut_method,
        mut_params=mut_params,
        xl=xl,
        xu=xu,
        workspace=variation_workspace,
        repair_cfg=cast(RepairConfigValue, cfg.get("repair", "auto")),
        problem=problem,
        variation_kernels=variation_kernels,
    )

    return variation


__all__ = [
    "build_variation_pipeline",
]
