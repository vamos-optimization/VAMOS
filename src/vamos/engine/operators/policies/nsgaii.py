# operators/policies/nsgaii.py
"""
Operator pool building for NSGA-II.

This module handles the construction of variation pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from vamos.engine.operators.impl.real import VariationWorkspace
from vamos.engine.variation import VariationPipeline
from vamos.engine.variation.protocol import CrossoverName, MutationName, RepairConfigValue
from vamos.foundation.encoding import EncodingLike
from vamos.foundation.problem.types import ProblemProtocol

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import VariationKernels


def build_operator_pool(
    cfg: dict[str, Any],
    encoding: EncodingLike,
    cross_method: str,
    cross_params: dict[str, Any],
    mut_method: str,
    mut_params: dict[str, Any],
    n_var: int,
    xl: NDArray[np.generic],
    xu: NDArray[np.generic],
    variation_workspace: VariationWorkspace,
    problem: ProblemProtocol,
    mut_factor: float | None,
    *,
    variation_kernels: VariationKernels | None = None,
) -> list[VariationPipeline]:
    """Build the operator pool.

    Parameters
    ----------
    cfg : dict[str, Any]
        Algorithm configuration dict.
    encoding : str
        Problem encoding type.
    cross_method : str
        Default crossover method.
    cross_params : dict[str, Any]
        Default crossover parameters.
    mut_method : str
        Default mutation method.
    mut_params : dict[str, Any]
        Default mutation parameters.
    n_var : int
        Number of decision variables.
    xl : NDArray[np.generic]
        Lower bounds.
    xu : NDArray[np.generic]
        Upper bounds.
    variation_workspace : VariationWorkspace
        Shared workspace for variation operators.
    problem : ProblemProtocol
        The optimization problem.
    mut_factor : float | None
        Optional mutation probability factor.
    variation_kernels : VariationKernels | None
        Optional backend kernels for SBX and polynomial mutation arithmetic.

    Returns:
        operator_pool
    """
    return _build_variation_pipelines(
        cfg,
        encoding,
        cross_method,
        cross_params,
        mut_method,
        mut_params,
        n_var,
        xl,
        xu,
        variation_workspace,
        problem,
        mut_factor,
        variation_kernels=variation_kernels,
    )


def _build_variation_pipelines(
    cfg: dict[str, Any],
    encoding: EncodingLike,
    cross_method: str,
    cross_params: dict[str, Any],
    mut_method: str,
    mut_params: dict[str, Any],
    n_var: int,
    xl: NDArray[np.generic],
    xu: NDArray[np.generic],
    variation_workspace: VariationWorkspace,
    problem: ProblemProtocol,
    mut_factor: float | None,
    *,
    variation_kernels: VariationKernels | None = None,
) -> list[VariationPipeline]:
    """Build list of variation pipelines from config.

    Parameters
    ----------
    cfg : dict[str, Any]
        Algorithm configuration.
    encoding : str
        Problem encoding type.
    cross_method : str
        Default crossover method.
    cross_params : dict[str, Any]
        Default crossover parameters.
    mut_method : str
        Default mutation method.
    mut_params : dict[str, Any]
        Default mutation parameters.
    n_var : int
        Number of decision variables.
    xl : NDArray[np.generic]
        Lower bounds.
    xu : NDArray[np.generic]
        Upper bounds.
    variation_workspace : VariationWorkspace
        Shared workspace.
    problem : ProblemProtocol
        The optimization problem.
    mut_factor : float | None
        Optional mutation probability factor.
    variation_kernels : VariationKernels | None
        Optional backend kernels for SBX and polynomial mutation arithmetic.

    Returns
    -------
    list[VariationPipeline]
        List of configured variation pipelines.
    """
    operator_pool: list[VariationPipeline] = []

    # Default pipeline
    operator_pool.append(
        _create_variation_pipeline(
            encoding,
            cross_method,
            cross_params,
            mut_method,
            mut_params,
            xl,
            xu,
            variation_workspace,
            cfg.get("repair", "auto"),
            problem,
            variation_kernels=variation_kernels,
        )
    )

    return operator_pool


def _create_variation_pipeline(
    encoding: EncodingLike,
    cross_method: str,
    cross_params: dict[str, Any],
    mut_method: str,
    mut_params: dict[str, Any],
    xl: NDArray[np.generic],
    xu: NDArray[np.generic],
    workspace: VariationWorkspace,
    repair_cfg: Any,
    problem: ProblemProtocol,
    *,
    variation_kernels: VariationKernels | None = None,
) -> VariationPipeline:
    """Create a single variation pipeline.

    Parameters
    ----------
    encoding : str
        Problem encoding type.
    cross_method : str
        Crossover method name.
    cross_params : dict[str, Any]
        Crossover parameters.
    mut_method : str
        Mutation method name.
    mut_params : dict[str, Any]
        Mutation parameters.
    xl : NDArray[np.generic]
        Lower bounds.
    xu : NDArray[np.generic]
        Upper bounds.
    workspace : VariationWorkspace
        Shared workspace.
    repair_cfg : Any
        Repair configuration or "auto".
    problem : ProblemProtocol
        The optimization problem.
    variation_kernels : VariationKernels | None
        Optional backend kernels for SBX and polynomial mutation arithmetic.

    Returns
    -------
    VariationPipeline
        Configured variation pipeline.
    """
    return VariationPipeline(
        encoding=encoding,
        cross_method=cast(CrossoverName, str(cross_method).lower()),
        cross_params=cross_params,
        mut_method=cast(MutationName, str(mut_method).lower()),
        mut_params=mut_params,
        xl=xl,
        xu=xu,
        workspace=workspace,
        repair_cfg=cast(RepairConfigValue, repair_cfg),
        problem=problem,
        variation_kernels=variation_kernels,
    )
