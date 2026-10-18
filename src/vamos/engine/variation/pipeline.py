"""
VariationPipeline class leveraging shared registries/helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import NDArray

from vamos.engine.operators.impl.registry import get_operator_registry
from vamos.engine.variation.helpers import (
    validate_operator_support,
)
from vamos.engine.variation.protocol import (
    CrossoverName,
    MutationName,
    RepairConfigValue,
    RepairName,
    RepairOperator,
    VariationWorkspaceProtocol,
)
from vamos.engine.variation.strategies import VariationContext, make_encoding_strategy
from vamos.foundation.encoding import EncodingLike, normalize_encoding
from vamos.foundation.problem.types import ProblemProtocol

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import VariationKernels


class VariationPipeline:
    """
    Encapsulates crossover + mutation (+ optional repair) for a given encoding.
    Builds encoding-specific operators that share a common call signature.
    """

    def __init__(
        self,
        *,
        encoding: EncodingLike,
        cross_method: CrossoverName,
        cross_params: dict[str, Any],
        mut_method: MutationName,
        mut_params: dict[str, Any],
        xl: NDArray[np.generic],
        xu: NDArray[np.generic],
        workspace: VariationWorkspaceProtocol | None,
        repair_cfg: RepairConfigValue = "auto",
        problem: ProblemProtocol | None = None,
        variation_kernels: VariationKernels | None = None,
    ) -> None:
        self.encoding = normalize_encoding(encoding)
        self.cross_method = cross_method
        self.cross_params = cross_params
        self.mut_method = mut_method
        self.mut_params = mut_params
        self.xl = xl
        self.xu = xu
        self.workspace = workspace
        self.problem = problem
        self.repair_cfg = repair_cfg

        validate_operator_support(self.encoding, cross_method, mut_method)

        ctx = VariationContext(
            xl=xl,
            xu=xu,
            workspace=workspace,
            problem=problem,
            variation_kernels=variation_kernels,
        )
        strategy = make_encoding_strategy(self.encoding, ctx)
        self.parents_per_group = strategy.parents_per_group(cross_method)
        self.children_per_group = strategy.children_per_group(cross_method)

        # Build operators
        self.crossover_op = strategy.build_crossover(cross_method, cross_params)
        self.mutation_op = strategy.build_mutation(mut_method, mut_params)
        self.repair_op = self._resolve_repair()

    def _resolve_repair(self) -> RepairOperator | None:
        if self.encoding != "real":
            if self.repair_cfg == "auto":
                return None
            raise ValueError("Repair operators are only supported for real encoding.")
        if self.repair_cfg == "auto":
            method: RepairName = "clip"
            params: dict[str, Any] = {}
        else:
            method, params = self.repair_cfg
        try:
            op_cls = cast(type[Any], get_operator_registry().get(method.lower()))
        except KeyError as exc:
            available = ", ".join(get_operator_registry().list())
            raise ValueError(f"Unknown repair operator '{method}'. Available: {available}") from exc
        try:
            op = op_cls(**params) if params else op_cls()
        except TypeError as exc:
            raise ValueError(f"Failed to initialize repair '{method}' with params {params}. Error: {exc}") from exc
        return cast(RepairOperator, op)

    def gather_parents(self, population: NDArray[np.generic], parent_idx: NDArray[np.integer[Any]]) -> NDArray[np.generic]:
        if self.workspace is None:
            return cast(NDArray[np.generic], population[parent_idx])
        shape = (parent_idx.size, population.shape[1])
        buffer = self.workspace.request("parent_buffer", shape, population.dtype)
        np.take(population, parent_idx, axis=0, out=buffer)
        return buffer

    def produce_offspring(self, parents: NDArray[np.generic], rng: np.random.Generator) -> NDArray[np.generic]:
        # Crossover
        offspring = self.crossover_op(parents, rng)
        if self.encoding == "real":
            assert self.repair_op is not None
            offspring = self.repair_op(offspring, self.xl, self.xu, rng)

        # Mutation
        offspring = self.mutation_op(offspring, rng)

        # Repair
        if self.repair_op is not None:
            offspring = self.repair_op(offspring, self.xl, self.xu, rng)

        return offspring


__all__ = ["VariationPipeline"]
