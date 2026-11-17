"""The flat view of a grammar that the current tuners sample: an :class:`AlgorithmConfigSpace`.

Each production becomes a conditional block of the parameter that it belongs to, and the global
sub-parameters of a parameter share its block, so a parameter of the view is active exactly when it
is active in the grammar. The view has no condition expressions.
"""

from __future__ import annotations

from ..racing.config_space import AlgorithmConfigSpace
from ..racing.param_space import Boolean, Categorical, ConditionalBlock, Int, ParamType, Real
from .model import Parameter, ParameterSpace


def to_algorithm_config_space(space: ParameterSpace) -> AlgorithmConfigSpace:
    """``space`` as an ``AlgorithmConfigSpace`` named after it, with its parameters in depth-first order."""
    params: list[ParamType] = []
    blocks: list[ConditionalBlock] = []

    def place(parameter: Parameter, holder: list[ParamType]) -> None:
        holder.append(_flat(parameter))
        for child in parameter.global_sub_parameters:
            place(child, holder)
        for value, production in zip(parameter.values, parameter.productions):
            if production:
                block = ConditionalBlock(parameter.name, value, [])
                blocks.append(block)
                for child in production:
                    place(child, block.params)

    for parameter in space.parameters:
        place(parameter, params)
    return AlgorithmConfigSpace(algorithm_name=space.name, params=params, conditionals=blocks or None)


def _flat(parameter: Parameter) -> ParamType:
    if parameter.type == "categorical":
        return Categorical(parameter.name, list(parameter.values), role=parameter.role or "operator")
    if parameter.type == "boolean":
        return Boolean(parameter.name, role=parameter.role or "adaptive")
    assert parameter.range is not None
    low, high = parameter.range
    if parameter.type == "int":
        return Int(parameter.name, int(low), int(high), log=parameter.log, role=parameter.role or "population")
    return Real(parameter.name, float(low), float(high), log=parameter.log, role=parameter.role or "operator_rate")


__all__ = ["to_algorithm_config_space"]
