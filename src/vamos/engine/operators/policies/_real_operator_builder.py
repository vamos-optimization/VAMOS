"""Shared typed helpers for real-valued policy operator construction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray

from vamos.engine.operators.impl.real import VariationWorkspace
from vamos.engine.operators.impl.real.utils import with_variation_kernels
from vamos.engine.operators.impl.registry import get_operator_registry
from vamos.engine.operators.policies.real_repair import apply_policy_repair, resolve_policy_repair
from vamos.engine.variation.helpers import resolve_prob_expression, validate_pairwise_real_operators
from vamos.engine.variation.protocol import CrossoverOperator, MutationOperator, RepairConfigValue, RepairOperator

if TYPE_CHECKING:
    from vamos.foundation.kernel.backend import VariationKernels


def _resolve_operator_factory(method: str) -> Callable[..., object]:
    try:
        factory = get_operator_registry().get(method.lower())
    except KeyError as exc:
        available = ", ".join(get_operator_registry().list())
        raise ValueError(f"Unknown operator '{method}'. Available: {available}") from exc
    if not callable(factory):
        raise ValueError(f"Registered operator '{method}' is not callable.")
    return cast(Callable[..., object], factory)


def instantiate_operator(
    method: str,
    params: Mapping[str, object],
    *,
    label: str,
    variation_kernels: VariationKernels | None = None,
) -> Callable[..., object]:
    """Instantiate a registry-backed operator with consistent error handling."""
    factory = _resolve_operator_factory(method)
    kwargs = with_variation_kernels(factory, dict(params), variation_kernels)
    try:
        return cast(Callable[..., object], factory(**kwargs))
    except TypeError as exc:
        raise ValueError(f"Failed to initialize {label} '{method}' with params {kwargs}. Error: {exc}") from exc


class RepairedRealOperator:
    """A real-coded crossover or mutation followed by the policy repair.

    Calling it applies ``operator`` and then ``repair`` within ``xl``/``xu``,
    using the run generator unless another one is passed. The parts stay
    inspectable so that an algorithm can recognise an operator chain that a
    compiled shortcut reproduces exactly.
    """

    __slots__ = ("operator", "repair", "xl", "xu", "rng")

    def __init__(
        self,
        operator: CrossoverOperator | MutationOperator,
        repair: RepairOperator,
        xl: NDArray[np.generic],
        xu: NDArray[np.generic],
        rng: np.random.Generator,
    ) -> None:
        self.operator = operator
        self.repair = repair
        self.xl = xl
        self.xu = xu
        self.rng = rng

    def __call__(self, values: NDArray[np.generic], _rng: np.random.Generator | None = None) -> NDArray[np.generic]:
        rng = self.rng if _rng is None else _rng
        varied = np.asarray(self.operator(values, rng))
        return apply_policy_repair(self.repair, varied, self.xl, self.xu, rng)


def build_real_variation_pair(
    *,
    cross_method: str,
    cross_params: Mapping[str, object],
    mut_method: str,
    mut_params: Mapping[str, object],
    n_var: int,
    xl: NDArray[np.generic],
    xu: NDArray[np.generic],
    rng: np.random.Generator,
    repair_cfg: RepairConfigValue,
    algorithm_label: str,
    variation_kernels: VariationKernels | None = None,
) -> tuple[RepairedRealOperator, RepairedRealOperator]:
    """Build real-coded crossover and mutation callables for a two-parent mating scheme.

    Operators of other encodings and three-parent crossovers raise ``ValueError``.
    """
    validate_pairwise_real_operators(cross_method, mut_method, algorithm_label=algorithm_label)
    workspace = VariationWorkspace()
    repair_operator = resolve_policy_repair("real", repair_cfg)
    assert repair_operator is not None

    cross_kwargs = dict(cross_params)
    prob = cross_kwargs.pop("prob", None)
    cross_prob = 0.9 if prob is None else float(cast(float | int | str, prob))
    cross_kwargs.setdefault("prob_crossover", cross_prob)
    cross_kwargs.setdefault("allow_inplace", True)
    cross_kwargs.setdefault("lower", xl)
    cross_kwargs.setdefault("upper", xu)
    cross_kwargs.setdefault("workspace", workspace)
    crossover_operator = cast(
        CrossoverOperator,
        instantiate_operator(cross_method, cross_kwargs, label="crossover", variation_kernels=variation_kernels),
    )

    mut_kwargs = dict(mut_params)
    raw_prob = cast(float | int | str | None, mut_kwargs.pop("prob", None))
    mut_kwargs.setdefault(
        "prob_mutation",
        resolve_prob_expression(raw_prob, n_var, 1.0 / max(1, n_var)),
    )
    mut_kwargs.setdefault("lower", xl)
    mut_kwargs.setdefault("upper", xu)
    mut_kwargs.setdefault("workspace", workspace)
    mutation_operator = cast(
        MutationOperator,
        instantiate_operator(mut_method, mut_kwargs, label="mutation", variation_kernels=variation_kernels),
    )

    return (
        RepairedRealOperator(crossover_operator, repair_operator, xl, xu, rng),
        RepairedRealOperator(mutation_operator, repair_operator, xl, xu, rng),
    )


__all__ = ["RepairedRealOperator", "build_real_variation_pair", "instantiate_operator"]
