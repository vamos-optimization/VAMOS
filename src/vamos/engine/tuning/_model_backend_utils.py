from __future__ import annotations

import ast
from typing import Any

import numpy as np

from .racing.param_space import Boolean, Categorical, Int, ParamSpace, Real, _condition_parents, _dependency_order, _parent_name

_OPTUNA_SAMPLERS: dict[str, str] = {
    "tpe": "TPESampler",
    "cmaes": "CmaEsSampler",
    "random": "RandomSampler",
    "nsgaii": "NSGAIISampler",
    "nsgaiii": "NSGAIIISampler",
    "qmc": "QMCSampler",
    "gp": "GPSampler",
}


def sample_from_optuna_trial(trial: Any, param_space: ParamSpace) -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    for name in _dependency_order(param_space):
        if not param_space.is_active(name, cfg):
            continue
        spec = param_space.params[name]
        if isinstance(spec, Real):
            cfg[name] = float(trial.suggest_float(name, float(spec.low), float(spec.high), log=bool(spec.log)))
        elif isinstance(spec, Int):
            cfg[name] = int(trial.suggest_int(name, int(spec.low), int(spec.high), log=bool(spec.log)))
        elif isinstance(spec, Categorical):
            cfg[name] = trial.suggest_categorical(name, list(spec.choices))
        elif isinstance(spec, Boolean):
            cfg[name] = bool(trial.suggest_categorical(name, [False, True]))
        else:  # pragma: no cover
            raise TypeError(f"Unsupported param spec type for '{name}': {type(spec)!r}")
    return cfg


def _configspace_condition(node: ast.AST, child: Any, cs: Any, *, negate: bool = False) -> Any:
    """Translate the supported condition AST without executing expression text."""
    from ConfigSpace.conditions import (
        AndConjunction,
        EqualsCondition,
        GreaterThanCondition,
        LessThanCondition,
        NotEqualsCondition,
        OrConjunction,
    )

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _configspace_condition(node.operand, child, cs, negate=not negate)
    if isinstance(node, ast.BoolOp):
        use_and = isinstance(node.op, ast.And) != negate
        conjunction = AndConjunction if use_and else OrConjunction
        return conjunction(*[_configspace_condition(value, child, cs, negate=negate) for value in node.values])
    if isinstance(node, ast.Subscript):
        parent = cs[_parent_name(node)]
        choices = getattr(parent, "choices", ())
        if not choices or not all(isinstance(value, (bool, np.bool_)) for value in choices):
            raise ValueError("Bare condition parents must be boolean parameters.")
        return EqualsCondition(child, parent, not negate)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.comparators[0], ast.Constant):
        parent = cs[_parent_name(node.left)]
        value = node.comparators[0].value
        op = type(node.ops[0])
        if negate:
            inverse = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.GtE, ast.LtE: ast.Gt, ast.Gt: ast.LtE, ast.GtE: ast.Lt}
            op = inverse.get(op, op)
        factories = {ast.Eq: EqualsCondition, ast.NotEq: NotEqualsCondition, ast.Lt: LessThanCondition, ast.Gt: GreaterThanCondition}
        if op in factories:
            return factories[op](child, parent, value)
        if op in (ast.LtE, ast.GtE):
            strict = LessThanCondition if op is ast.LtE else GreaterThanCondition
            return OrConjunction(strict(child, parent, value), EqualsCondition(child, parent, value))
    raise ValueError("Expected a comparison of cfg['parent'] with a scalar literal, a boolean parent, or and/or/not.")


def build_configspace(param_space: ParamSpace, seed: int) -> Any:
    from ConfigSpace import ConfigurationSpace
    from ConfigSpace.conditions import AndConjunction
    from ConfigSpace.hyperparameters import (
        CategoricalHyperparameter,
        UniformFloatHyperparameter,
        UniformIntegerHyperparameter,
    )

    names = _dependency_order(param_space)
    cs = ConfigurationSpace(seed=int(seed))
    for name in names:
        spec = param_space.params[name]
        hp: Any
        if isinstance(spec, Real):
            hp = UniformFloatHyperparameter(name=name, lower=float(spec.low), upper=float(spec.high), log=bool(spec.log))
        elif isinstance(spec, Int):
            hp = UniformIntegerHyperparameter(name=name, lower=int(spec.low), upper=int(spec.high), log=bool(spec.log))
        elif isinstance(spec, Categorical):
            hp = CategoricalHyperparameter(name=name, choices=list(spec.choices))
        elif isinstance(spec, Boolean):
            hp = CategoricalHyperparameter(name=name, choices=[False, True])
        else:  # pragma: no cover
            raise TypeError(f"Unsupported param spec type for '{name}': {type(spec)!r}")
        cs.add(hp)
    expressions: dict[str, list[str]] = {name: [] for name in names}
    for condition in param_space.conditions:
        expressions[condition.param_name].append(condition.expr)
    for name in names:
        own_expressions = expressions[name]
        # VAMOS treats a missing parent as inactive, even inside an OR or NOT.
        # Native OR conditions also need the parents' activation guards.
        inherited = [expr for own in own_expressions for parent in _condition_parents(own) for expr in expressions[parent]]
        expressions[name] = list(dict.fromkeys([*own_expressions, *inherited]))
        native_conditions = []
        for expr in expressions[name]:
            try:
                native_conditions.append(_configspace_condition(ast.parse(expr, mode="eval").body, cs[name], cs))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Cannot translate condition for '{name}' to ConfigSpace: {expr!r}. {exc}") from exc
        if native_conditions:
            cs.add(native_conditions[0] if len(native_conditions) == 1 else AndConjunction(*native_conditions))
    return cs


def hyperband_brackets(max_sh_iter: int, eta: int, max_configs: int) -> list[list[int]]:
    """Return the configurations per rung of each HpBandSter BOHB iteration, ``max_configs`` in all.

    Iteration ``i`` is HpBandSter's bracket (``BOHB.get_next_iteration``): ``s = max_sh_iter - 1 -
    i % max_sh_iter`` halvings from ``floor(max_sh_iter / (s + 1)) * eta**s`` configurations. The
    last bracket starts with the configurations left, so the search samples exactly ``max_configs``.
    """
    brackets: list[list[int]] = []
    remaining = max(1, int(max_configs))
    while remaining > 0:
        s = max_sh_iter - 1 - len(brackets) % max_sh_iter
        n0 = min(remaining, int(np.floor(max_sh_iter / (s + 1)) * eta**s))
        brackets.append([max(int(n0 * eta ** (-i)), 1) for i in range(s + 1)])
        remaining -= n0
    return brackets


def build_optuna_sampler(name: str, seed: int) -> Any:
    import optuna.samplers as samplers

    key = name.lower().replace("-", "").replace("_", "")
    cls_name = _OPTUNA_SAMPLERS.get(key)
    if cls_name is None:
        supported = ", ".join(sorted(_OPTUNA_SAMPLERS))
        raise ValueError(f"Unknown optuna_sampler {name!r}. Choose from: {supported}")
    cls = getattr(samplers, cls_name)
    if key == "tpe":
        return cls(seed=seed, multivariate=True, group=True)
    return cls(seed=seed)
