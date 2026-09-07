"""
Algorithm configuration space with conditional parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .param_space import PARAM_ROLES, Condition, ParamSpace, ParamType
from .parameters import ConditionalBlock

# Type alias for composable space parts: (params, conditionals, conditions).
SpacePart = tuple[list[ParamType], list[ConditionalBlock], list[Condition]]


@dataclass
class AlgorithmConfigSpace:
    """
    Declarative hyperparameter space for an algorithm.
    """

    algorithm_name: str
    params: list[ParamType]
    conditionals: list[ConditionalBlock] | None = None
    conditions: list[Condition] | None = None

    def _conditional_params(self) -> list[ParamType]:
        params: list[ParamType] = []
        seen = {p.name for p in self.params}
        for block in self.conditionals or []:
            for p in block.params:
                if p.name in seen:
                    continue
                params.append(p)
                seen.add(p.name)
        return params

    def _combined_conditions(self) -> list[Condition]:
        conditions: list[Condition] = []
        for block in self.conditionals or []:
            expr = f"cfg['{block.parent_name}'] == {block.parent_value!r}"
            for p in block.params:
                conditions.append(Condition(p.name, expr))
        if self.conditions:
            conditions.extend(self.conditions)
        return conditions

    def flatten(self, assignment: dict[str, Any] | None = None) -> list[ParamType]:
        """
        Return the list of active parameters given a partial assignment.
        If assignment is None, only top-level params are returned.
        """
        if assignment is None:
            return list(self.params)
        param_space = self.to_param_space()
        active = [p for p in self.params if param_space.is_active(p.name, assignment)]
        for p in self._conditional_params():
            if param_space.is_active(p.name, assignment):
                active.append(p)
        return active

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        """
        Sample a concrete assignment dict for all active params.
        """
        assignment: dict[str, Any] = {}
        param_space = self.to_param_space()
        for p in self.params:
            if param_space.is_active(p.name, assignment):
                assignment[p.name] = p.sample(rng)
        for p in self._conditional_params():
            if param_space.is_active(p.name, assignment):
                assignment[p.name] = p.sample(rng)
        return assignment

    def to_unit_vector(self, assignment: dict[str, Any]) -> NDArray[np.float64]:
        """
        Encode an assignment into a unit vector [0,1]^D following param order.
        """
        values: list[float] = []
        param_space = self.to_param_space()
        for p in self.params:
            if param_space.is_active(p.name, assignment):
                values.append(float(p.to_unit(assignment[p.name])))
        for p in self._conditional_params():
            if param_space.is_active(p.name, assignment):
                values.append(float(p.to_unit(assignment[p.name])))
        return np.asarray(values, dtype=float)

    def from_unit_vector(self, u: NDArray[np.float64]) -> dict[str, Any]:
        """
        Decode a unit vector into an assignment dict, respecting conditionals.
        """
        assignment: dict[str, Any] = {}
        idx = 0
        param_space = self.to_param_space()
        for p in self.params:
            if param_space.is_active(p.name, assignment):
                assignment[p.name] = p.from_unit(float(u[idx]))
                idx += 1
        for p in self._conditional_params():
            if param_space.is_active(p.name, assignment):
                assignment[p.name] = p.from_unit(float(u[idx]))
                idx += 1
        return assignment

    def group_by_role(self) -> dict[str, list[ParamType]]:
        """Group all parameters (including conditionals) by their role.

        Returns a dict mapping role name to list of parameters with that role.
        Useful for hierarchical tuning where different roles are tuned in phases.
        """
        groups: dict[str, list[ParamType]] = {role: [] for role in PARAM_ROLES}
        for p in self.params:
            groups.setdefault(p.role, []).append(p)
        for p in self._conditional_params():
            groups.setdefault(p.role, []).append(p)
        return {k: v for k, v in groups.items() if v}

    def to_param_space(self) -> ParamSpace:
        """
        Convert this config space into a ParamSpace suitable for the racing pipeline.
        """
        params: dict[str, ParamType] = {p.name: p for p in self.params}
        conditions = self._combined_conditions()

        for block in self.conditionals or []:
            for p in block.params:
                if p.name in params and params[p.name] is not p:
                    raise ValueError(f"Duplicate parameter '{p.name}' in conditional blocks.")
                params[p.name] = p

        if self.conditions:
            for cond in self.conditions:
                if cond.param_name not in params:
                    raise ValueError(f"Condition references unknown parameter '{cond.param_name}'.")

        return ParamSpace(params=params, conditions=conditions)


def compose_config_space(algorithm_name: str, *parts: SpacePart) -> AlgorithmConfigSpace:
    """
    Compose an :class:`AlgorithmConfigSpace` from multiple
    ``(params, conditionals, conditions)`` parts.

    This enables separating algorithm-specific parameters from
    encoding-specific operator parameters and combining them cleanly::

        def build_moead_mixed_config_space():
            return compose_config_space(
                "moead_mixed",
                _moead_core_part(),
                _moead_aggregation_part(),
                _mixed_operator_part(),
            )
    """
    all_params: list[ParamType] = []
    all_conditionals: list[ConditionalBlock] = []
    all_conditions: list[Condition] = []
    for params, conditionals, conditions in parts:
        all_params.extend(params)
        all_conditionals.extend(conditionals)
        all_conditions.extend(conditions)
    return AlgorithmConfigSpace(
        algorithm_name=algorithm_name,
        params=all_params,
        conditionals=all_conditionals or None,
        conditions=all_conditions or None,
    )
