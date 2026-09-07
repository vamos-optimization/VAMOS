"""
Hyperparameter space definitions for VAMOS tuning.

All parameter types use name as the first argument:
- Real(name, low, high, log=False, role="operator_rate")
- Int(name, low, high, log=False, role="population")
- Categorical(name, choices, role="operator")
- Boolean(name, role="adaptive")

All types support:
- sample(rng) - draw a random value
- to_unit(value) - map to [0, 1] space for optimization
- from_unit(value) - map from [0, 1] space back to parameter value

Parameter roles (used by structured tuning workflows):
- "structural": algorithm paradigm choices (decomposition, reference points)
- "operator": discrete operator family choices (crossover type, mutation type)
- "operator_rate": continuous operator parameters (eta, probabilities, sigma)
- "population": resource allocation (pop_size, offspring_ratio, archive_size)
- "adaptive": meta-parameters (immigration, adaptive operator selection)
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Real:
    """Real-valued hyperparameter in [low, high]."""

    name: str
    low: float
    high: float
    log: bool = False
    role: str = "operator_rate"

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"Real param '{self.name}' has high < low ({self.high} < {self.low}).")
        if self.log and (self.low <= 0 or self.high <= 0):
            raise ValueError(f"Real param '{self.name}' with log=True requires low/high > 0.")

    def sample(self, rng: np.random.Generator) -> float:
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            return float(math.exp(rng.uniform(lo, hi)))
        return float(rng.uniform(self.low, self.high))

    def to_unit(self, value: float) -> float:
        """Map value to [0, 1] space."""
        v = float(value)
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            return 0.0 if hi == lo else (math.log(v) - lo) / (hi - lo)
        return 0.0 if self.high == self.low else (v - self.low) / (self.high - self.low)

    def from_unit(self, value: float) -> float:
        """Map from [0, 1] space to parameter value."""
        u = min(max(value, 0.0), 1.0)
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            return float(math.exp(lo + u * (hi - lo)))
        return float(self.low + u * (self.high - self.low))


@dataclass
class Int:
    """Integer hyperparameter in [low, high] (inclusive)."""

    name: str
    low: int
    high: int
    log: bool = False
    role: str = "population"

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"Int param '{self.name}' has high < low ({self.high} < {self.low}).")
        if self.log and (self.low <= 0 or self.high <= 0):
            raise ValueError(f"Int param '{self.name}' with log=True requires low/high > 0.")

    def sample(self, rng: np.random.Generator) -> int:
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            return int(round(math.exp(rng.uniform(lo, hi))))
        return int(rng.integers(self.low, self.high + 1))

    def to_unit(self, value: int) -> float:
        """Map value to [0, 1] space."""
        v = float(value)
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            return 0.0 if hi == lo else (math.log(v) - lo) / (hi - lo)
        return 0.0 if self.high == self.low else (v - self.low) / (self.high - self.low)

    def from_unit(self, value: float) -> int:
        """Map from [0, 1] space to parameter value."""
        u = min(max(value, 0.0), 1.0)
        if self.log:
            lo, hi = math.log(self.low), math.log(self.high)
            mapped = math.exp(lo + u * (hi - lo))
        else:
            mapped = self.low + u * (self.high - self.low)
        return int(round(mapped))


@dataclass
class Categorical:
    """Categorical hyperparameter with discrete choices."""

    name: str
    choices: Sequence[Any]
    role: str = "operator"

    def sample(self, rng: np.random.Generator) -> Any:
        return self.choices[int(rng.integers(0, len(self.choices)))]

    def to_unit(self, value: Any) -> float:
        """Map value to [0, 1] space based on choice index."""
        idx = list(self.choices).index(value)
        return 0.0 if len(self.choices) == 1 else idx / float(len(self.choices) - 1)

    def from_unit(self, value: float) -> Any:
        """Map from [0, 1] space to a choice."""
        u = min(max(value, 0.0), 1.0)
        idx = int(round(u * (len(self.choices) - 1)))
        return self.choices[idx]


@dataclass
class Boolean:
    """Boolean hyperparameter (True/False)."""

    name: str
    role: str = "adaptive"

    def sample(self, rng: np.random.Generator) -> bool:
        return bool(rng.integers(0, 2))

    def to_unit(self, value: bool) -> float:
        return 1.0 if value else 0.0

    def from_unit(self, value: float) -> bool:
        return value >= 0.5


@dataclass
class Condition:
    """
    Simple condition: a parameter is considered active only when
    `expr` evaluates to True given the current config.

    Expressions are parsed safely (no function calls/attribute access).
    """

    param_name: str
    expr: str  # Python expression using a dict `cfg`


@dataclass
class ConditionalBlock:
    """
    A block of parameters that are active only when a parent parameter
    has a specific value.
    """

    parent_name: str
    parent_value: Any
    params: list[ParamType]  # list of Real, Int, Categorical, etc.


# Type alias for any parameter type
ParamType = Real | Int | Categorical | Boolean

# Valid parameter roles for structured tuning workflows
PARAM_ROLES = {"structural", "operator", "operator_rate", "population", "adaptive"}


@dataclass
class ParamSpace:
    """
    Defines a hyperparameter space with named parameters.

    Example:
        space = ParamSpace(params={
            "lr": Real("lr", 0.001, 0.1, log=True),
            "epochs": Int("epochs", 10, 100),
        })
    """

    params: dict[str, ParamType] = field(default_factory=dict)
    conditions: list[Condition] = field(default_factory=list)

    def sample(self, rng: np.random.Generator | None = None) -> dict[str, Any]:
        """Sample a configuration from the space."""
        rng = np.random.default_rng() if rng is None else rng
        full_cfg = {name: spec.sample(rng) for name, spec in self.params.items()}
        return {name: value for name, value in full_cfg.items() if self.is_active(name, full_cfg)}

    def is_active(self, param_name: str, config: dict[str, Any]) -> bool:
        """Check if param is active given config and conditions."""
        relevant = [c for c in self.conditions if c.param_name == param_name]
        if not relevant:
            return True
        return all(_safe_eval_condition(c.expr, config) for c in relevant)

    def validate(self, config: dict[str, Any]) -> None:
        """Validate that all active params are present and within bounds."""
        for name, spec in self.params.items():
            if not self.is_active(name, config):
                continue
            if name not in config:
                raise ValueError(f"Active parameter '{name}' missing from config")

            value = config[name]
            if isinstance(spec, Real):
                if not (spec.low <= value <= spec.high):
                    raise ValueError(f"Real param '{name}'={value} out of [{spec.low}, {spec.high}]")
            elif isinstance(spec, Int):
                if not (spec.low <= value <= spec.high):
                    raise ValueError(f"Int param '{name}'={value} out of [{spec.low}, {spec.high}]")
            elif isinstance(spec, Categorical):
                if value not in spec.choices:
                    raise ValueError(f"Categorical param '{name}'={value} not in {spec.choices}")


# Aliases
FloatParam = Real
IntegerParam = Int
CategoricalParam = Categorical
BooleanParam = Boolean
CategoricalIntegerParam = Categorical


_CMP_OPS: dict[type[ast.AST], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


class _MissingKeyError(Exception):
    """Raised when a condition references a cfg key that is absent (inactive parent)."""


def _safe_eval_condition(expr: str, cfg: dict[str, Any]) -> bool:
    """
    Evaluate a condition expression against cfg using a restricted AST (no calls/attrs).

    If the expression references a cfg key that is missing (because its parent
    parameter is inactive), the condition evaluates to ``False``.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - validated upstream
        raise ValueError(f"Invalid condition syntax: {expr}") from exc

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.BoolOp):
            values = [_eval(v) for v in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            if isinstance(node.op, ast.Or):
                return any(values)
            raise ValueError(f"Unsupported boolean operator in: {expr}")
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not _eval(node.operand)
        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            for op_node, comparator in zip(node.ops, node.comparators):
                rhs = _eval(comparator)
                op_fn = _CMP_OPS.get(type(op_node))
                if op_fn is None:
                    raise ValueError(f"Unsupported comparison in: {expr}")
                if not op_fn(left, rhs):
                    return False
                left = rhs
            return True
        if isinstance(node, ast.Name):
            if node.id != "cfg":
                raise ValueError(f"Only 'cfg' is allowed in conditions (got '{node.id}').")
            return cfg
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Subscript):
            base = _eval(node.value)
            key = _eval(node.slice)
            try:
                return base[key]
            except KeyError as exc:
                raise _MissingKeyError(key) from exc
            except Exception as exc:
                raise ValueError(f"Failed to access cfg[{key!r}] in condition: {expr}") from exc
        raise ValueError(f"Unsupported expression element in condition: {expr}")

    try:
        result = _eval(tree)
    except _MissingKeyError:
        return False
    if not isinstance(result, (bool, np.bool_)):
        raise ValueError(f"Condition must evaluate to bool, got {result!r} for: {expr}")
    return bool(result)


__all__ = [
    "ParamSpace",
    "Real",
    "Int",
    "Categorical",
    "Boolean",
    "Condition",
    "ConditionalBlock",
    "ParamType",
    "PARAM_ROLES",
    # Aliases
    "FloatParam",
    "IntegerParam",
    "CategoricalParam",
    "BooleanParam",
    "CategoricalIntegerParam",
]
