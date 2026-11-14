"""Internal builders for the algorithms designed for constrained problems."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from vamos.engine.algorithm._builder_common import (
    apply_operator_config,
    apply_optional_operator_config,
    finalize_algorithm,
    resolve_problem_encoding,
    resolve_variation_config,
)
from vamos.engine.algorithm.config import AlgorithmConfigProtocol, CCMOConfig
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.problem.types import ProblemProtocol

if TYPE_CHECKING:
    from vamos.engine.algorithm.ccmo import CCMO


def build_ccmo_algorithm(
    *,
    kernel: KernelBackend,
    problem: ProblemProtocol,
    pop_size: int,
    ccmo_variation: dict[str, Any] | None,
) -> tuple[CCMO, AlgorithmConfigProtocol]:
    """CCMO with the paper's variation for real variables, SBX(1.0, 20) and polynomial mutation (1/n, 20), unless overridden."""
    encoding = resolve_problem_encoding(problem)
    ccmo_overrides = ccmo_variation or {}
    real_defaults = {
        "crossover": ("sbx", {"prob": 1.0, "eta": 20.0}),
        "mutation": ("polynomial", {"prob": 1.0 / problem.n_var, "eta": 20.0}),
    }
    var_cfg = resolve_variation_config(encoding=encoding, overrides=ccmo_overrides, defaults=real_defaults if encoding == "real" else None)

    builder = CCMOConfig.builder()
    builder.pop_size(pop_size)
    apply_operator_config(builder, method_name="crossover", operator_cfg=var_cfg["crossover"], key="crossover")
    apply_operator_config(builder, method_name="mutation", operator_cfg=var_cfg["mutation"], key="mutation")
    apply_optional_operator_config(
        builder,
        method_name="repair",
        operator_cfg=var_cfg.get("repair"),
        key="repair",
        encoding=encoding,
    )
    return cast(
        "tuple[CCMO, AlgorithmConfigProtocol]",
        finalize_algorithm(algorithm_name="ccmo", builder=builder, kernel=kernel, block=ccmo_overrides),
    )


__all__ = ["build_ccmo_algorithm"]
