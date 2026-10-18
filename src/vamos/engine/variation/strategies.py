"""
Encoding-specific variation wiring.

This module isolates encoding differences (operator registries + call signatures)
behind a small strategy interface so `VariationPipeline` stays simple and
extensible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from vamos.engine.operators.impl.real.utils import with_variation_kernels
from vamos.engine.operators.impl.registry import get_operator_registry
from vamos.engine.variation.helpers import (
    BINARY_CROSSOVER,
    BINARY_MUTATION,
    INT_CROSSOVER,
    INT_MUTATION,
    MIXED_CROSSOVER,
    MIXED_MUTATION,
    PERM_CROSSOVER,
    PERM_MUTATION,
)
from vamos.engine.variation.protocol import (
    CrossoverName,
    CrossoverOperator,
    MutationName,
    MutationOperator,
    VariationWorkspaceProtocol,
)
from vamos.foundation.encoding import Encoding
from vamos.foundation.problem.types import ProblemProtocol

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import VariationKernels


@dataclass(frozen=True)
class VariationContext:
    xl: NDArray[np.generic]
    xu: NDArray[np.generic]
    workspace: VariationWorkspaceProtocol | None
    problem: ProblemProtocol | None
    variation_kernels: VariationKernels | None = None


class EncodingStrategy(Protocol):
    @property
    def encoding(self) -> Encoding: ...

    def parents_per_group(self, cross_method: CrossoverName) -> int: ...

    def children_per_group(self, cross_method: CrossoverName) -> int: ...

    def build_crossover(self, method: CrossoverName, params: dict[str, Any]) -> CrossoverOperator: ...

    def build_mutation(self, method: MutationName, params: dict[str, Any]) -> MutationOperator: ...


@dataclass(frozen=True)
class RealEncodingStrategy:
    ctx: VariationContext
    encoding: Encoding = "real"

    def parents_per_group(self, cross_method: CrossoverName) -> int:
        return 3 if cross_method in {"pcx", "undx", "simplex", "de"} else 2

    def children_per_group(self, cross_method: CrossoverName) -> int:
        if cross_method == "de":
            return 1
        if cross_method == "undx":
            return 2
        if cross_method in {"pcx", "simplex"}:
            return 3
        return 2

    def build_crossover(self, method: CrossoverName, params: dict[str, Any]) -> CrossoverOperator:
        registry = get_operator_registry()
        try:
            op_cls = cast(type[Any], registry.get(method))
        except KeyError as exc:
            available = ", ".join(registry.list())
            raise ValueError(f"Unknown real crossover '{method}'. Available: {available}") from exc

        kwargs = dict(params)
        prob = kwargs.pop("prob", None)
        if prob is not None and "prob_crossover" in kwargs:
            raise ValueError("Use either 'prob' or 'prob_crossover', not both.")
        kwargs.setdefault("prob_crossover", 0.9 if prob is None else float(prob))
        kwargs.setdefault("allow_inplace", True)
        kwargs.setdefault("lower", self.ctx.xl)
        kwargs.setdefault("upper", self.ctx.xu)
        kwargs.setdefault("workspace", self.ctx.workspace)
        with_variation_kernels(op_cls, kwargs, self.ctx.variation_kernels)

        try:
            op = op_cls(**kwargs)
        except TypeError as exc:
            raise ValueError(f"Failed to initialize crossover '{method}' with params {kwargs}. Error: {exc}") from exc

        group_size = self.parents_per_group(method)

        def crossover(parents: NDArray[np.generic], rng: np.random.Generator) -> NDArray[np.generic]:
            n_var = parents.shape[1]
            if parents.shape[0] % group_size != 0:
                usable = (parents.shape[0] // group_size) * group_size
                parents = parents[:usable]
            if parents.size == 0:
                return parents

            groups = parents.reshape(-1, group_size, n_var)
            offspring = cast(NDArray[np.generic], op(groups, rng))
            if offspring.ndim == 3:
                offspring = offspring.reshape(-1, n_var)
            return offspring

        return crossover

    def build_mutation(self, method: MutationName, params: dict[str, Any]) -> MutationOperator:
        registry = get_operator_registry()
        try:
            op_cls = cast(type[Any], registry.get(method))
        except KeyError as exc:
            available = ", ".join(registry.list())
            raise ValueError(f"Unknown real mutation '{method}'. Available: {available}") from exc

        kwargs = dict(params)
        prob = kwargs.pop("prob", None)
        if prob is not None and "prob_mutation" in kwargs:
            raise ValueError("Use either 'prob' or 'prob_mutation', not both.")
        kwargs.setdefault("prob_mutation", 0.1 if prob is None else float(prob))
        kwargs.setdefault("lower", self.ctx.xl)
        kwargs.setdefault("upper", self.ctx.xu)
        kwargs.setdefault("workspace", self.ctx.workspace)
        with_variation_kernels(op_cls, kwargs, self.ctx.variation_kernels)

        try:
            op = op_cls(**kwargs)
        except TypeError as exc:
            raise ValueError(f"Failed to initialize mutation '{method}' with params {kwargs}. Error: {exc}") from exc

        def mutate(offspring: NDArray[np.generic], rng: np.random.Generator) -> NDArray[np.generic]:
            return cast(NDArray[np.generic], op(offspring, rng))

        return mutate


@dataclass(frozen=True)
class PairwiseEncodingStrategy:
    """
    Strategy for encodings with pairwise crossover and in-place mutation functions.

    Handles "binary" and "permutation" which share the same signatures:
    - crossover(X_parents, prob, rng) -> NDArray[np.generic]
    - mutation(X, prob, rng) -> None
    """

    ctx: VariationContext
    encoding: Encoding

    def parents_per_group(self, cross_method: CrossoverName) -> int:
        return 2

    def children_per_group(self, cross_method: CrossoverName) -> int:
        return 2

    def build_crossover(self, method: CrossoverName, params: dict[str, Any]) -> CrossoverOperator:
        registry = PERM_CROSSOVER if self.encoding == "permutation" else BINARY_CROSSOVER
        cross_prob = float(params.get("prob", 1.0))
        cross_fn = cast(Any, registry[method])

        def crossover(parents: NDArray[np.generic], rng: np.random.Generator) -> NDArray[np.generic]:
            return cast(NDArray[np.generic], cross_fn(parents, cross_prob, rng))

        return crossover

    def build_mutation(self, method: MutationName, params: dict[str, Any]) -> MutationOperator:
        registry = PERM_MUTATION if self.encoding == "permutation" else BINARY_MUTATION
        mut_prob = float(params.get("prob", 0.0))
        mut_fn = cast(Any, registry[method])

        def mutate(offspring: NDArray[np.generic], rng: np.random.Generator) -> NDArray[np.generic]:
            mut_fn(offspring, mut_prob, rng)
            return offspring

        return mutate


@dataclass(frozen=True)
class IntegerEncodingStrategy:
    ctx: VariationContext
    encoding: Encoding = "integer"

    def parents_per_group(self, cross_method: CrossoverName) -> int:
        return 2

    def children_per_group(self, cross_method: CrossoverName) -> int:
        return 2

    def build_crossover(self, method: CrossoverName, params: dict[str, Any]) -> CrossoverOperator:
        cross_prob = float(params.get("prob", 1.0))
        cross_fn = cast(Any, INT_CROSSOVER[method])
        if method == "sbx":
            eta = float(params.get("eta", 20.0))
            xl, xu = self.ctx.xl, self.ctx.xu
        else:
            eta = None
            xl, xu = self.ctx.xl, self.ctx.xu

        def crossover(parents: NDArray[np.generic], rng: np.random.Generator) -> NDArray[np.generic]:
            if eta is not None:
                return cast(NDArray[np.generic], cross_fn(parents, cross_prob, float(eta), xl, xu, rng))
            return cast(NDArray[np.generic], cross_fn(parents, cross_prob, rng))

        return crossover

    def build_mutation(self, method: MutationName, params: dict[str, Any]) -> MutationOperator:
        mut_prob = float(params.get("prob", 0.0))
        mut_fn = cast(Any, INT_MUTATION[method])
        xl, xu = self.ctx.xl, self.ctx.xu
        if method == "creep":
            step = int(params.get("step", 1))
            extra: float | int | None = step
        elif method in {"pm", "polynomial"}:
            extra = float(params.get("eta", 20.0))
        elif method == "gaussian":
            extra = float(params.get("sigma", 1.0))
        else:
            extra = None

        def mutate(offspring: NDArray[np.generic], rng: np.random.Generator) -> NDArray[np.generic]:
            if extra is not None:
                mut_fn(offspring, mut_prob, extra, xl, xu, rng)
            else:
                mut_fn(offspring, mut_prob, xl, xu, rng)
            return offspring

        return mutate


@dataclass(frozen=True)
class MixedEncodingStrategy:
    ctx: VariationContext
    encoding: Encoding = "mixed"

    def parents_per_group(self, cross_method: CrossoverName) -> int:
        return 2

    def children_per_group(self, cross_method: CrossoverName) -> int:
        return 2

    def _merged_spec(self, params: dict[str, Any]) -> dict[str, Any]:
        base_spec = getattr(self.ctx.problem, "mixed_spec", None)
        if base_spec is None:
            raise ValueError("Mixed-encoding problems must define 'mixed_spec'.")
        merged = dict(base_spec)
        for key, value in params.items():
            if key == "prob":
                continue
            merged[key] = value
        return merged

    def build_crossover(self, method: CrossoverName, params: dict[str, Any]) -> CrossoverOperator:
        spec = self._merged_spec(params)
        cross_prob = float(params.get("prob", 1.0))
        cross_fn = cast(Any, MIXED_CROSSOVER[method])

        def crossover(parents: NDArray[np.generic], rng: np.random.Generator) -> NDArray[np.generic]:
            return cast(NDArray[np.generic], cross_fn(parents, cross_prob, spec, rng))

        return crossover

    def build_mutation(self, method: MutationName, params: dict[str, Any]) -> MutationOperator:
        spec = self._merged_spec(params)
        mut_prob = float(params.get("prob", 0.0))
        mut_fn = cast(Any, MIXED_MUTATION[method])

        def mutate(offspring: NDArray[np.generic], rng: np.random.Generator) -> NDArray[np.generic]:
            mut_fn(offspring, mut_prob, spec, rng)
            return offspring

        return mutate


def make_encoding_strategy(encoding: Encoding, ctx: VariationContext) -> EncodingStrategy:
    if encoding == "real":
        return RealEncodingStrategy(ctx)
    if encoding == "binary":
        return PairwiseEncodingStrategy(ctx, encoding="binary")
    if encoding == "permutation":
        return PairwiseEncodingStrategy(ctx, encoding="permutation")
    if encoding == "integer":
        return IntegerEncodingStrategy(ctx)
    if encoding == "mixed":
        return MixedEncodingStrategy(ctx)
    raise ValueError(f"Unsupported encoding '{encoding}'.")


__all__ = [
    "EncodingStrategy",
    "VariationContext",
    "make_encoding_strategy",
]
