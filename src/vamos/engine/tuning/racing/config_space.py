"""
Algorithm configuration space with conditional parameters.

The built-in spaces are views of YAML grammars (:mod:`vamos.engine.tuning.grammar`); a space for
another algorithm can still be written directly as an ``AlgorithmConfigSpace``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .param_space import PARAM_ROLES, Condition, ParamSpace, ParamType, _dependency_order
from .parameters import ConditionalBlock

if TYPE_CHECKING:
    from ..grammar.model import ParameterSpace


@dataclass
class AlgorithmConfigSpace:
    """
    Declarative hyperparameter space for an algorithm.
    """

    algorithm_name: str
    params: list[ParamType]
    conditionals: list[ConditionalBlock] | None = None
    conditions: list[Condition] | None = None
    # The grammar that this space is a view of (built-in spaces); the views sample by walking its tree.
    grammar: ParameterSpace | None = None

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

    def _dependency_ordered(self, param_space: ParamSpace) -> list[ParamType]:
        """Top-level then conditional parameters, moving each after the parents its conditions read."""
        return [param_space.params[name] for name in _dependency_order(param_space)]

    def sample(self, rng: np.random.Generator) -> dict[str, Any]:
        """
        Sample a concrete assignment dict for all active params.
        """
        if self.grammar is not None:
            return dict(self.grammar.sample(rng))
        assignment: dict[str, Any] = {}
        param_space = self.to_param_space()
        for p in self._dependency_ordered(param_space):
            if param_space.is_active(p.name, assignment):
                assignment[p.name] = p.sample(rng)
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

        return ParamSpace(params=params, conditions=conditions, grammar=self.grammar)
