"""Translate a VAMOS ``ParamSpace`` to irace's parameter syntax and back.

irace receives one line per parameter, with its conditions after ``|``. Values travel on the
command line as ``--name=token``, so every categorical choice needs a distinct token made of
letters, digits, ``_``, ``.``, ``+`` or ``-``. Conditions follow the subset that the ConfigSpace
translation supports: comparisons of ``cfg['parent']`` with a scalar literal, boolean parents,
and ``and``/``or``/``not``.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .racing.param_space import (
    Boolean,
    Categorical,
    Int,
    ParamSpace,
    ParamType,
    Real,
    _active_assignment,
    _condition_parents,
    _dependency_order,
    _parent_name,
)

_NAME = r"[A-Za-z][A-Za-z0-9_]*"
_TOKEN = r"[A-Za-z0-9_.+-]+"
_R_OPS: dict[type[ast.cmpop], str] = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}


def _number(value: float) -> str:
    return repr(float(value)) if isinstance(value, float) else str(int(value))


def _tokens(name: str, spec: Categorical | Boolean) -> dict[str, Any]:
    """Map each irace token of a categorical or boolean parameter to its Python value."""
    if isinstance(spec, Boolean):
        return {"FALSE": False, "TRUE": True}
    tokens = {(choice if isinstance(choice, str) else repr(choice)): choice for choice in spec.choices}
    if len(tokens) != len(spec.choices) or not all(re.fullmatch(_TOKEN, token) for token in tokens):
        raise ValueError(f"Choices of '{name}' need distinct irace tokens made of letters, digits, '_', '.', '+' or '-'.")
    return tokens


def _token_of(name: str, spec: Categorical | Boolean, value: Any) -> str:
    for token, choice in _tokens(name, spec).items():
        # Python's True == 1; keep booleans and numbers apart as the choices do.
        if choice == value and isinstance(choice, bool) == isinstance(value, bool):
            return token
    raise ValueError(f"{value!r} is not a choice of '{name}'.")


def _r_expr(node: ast.AST, space: ParamSpace) -> str:
    if isinstance(node, ast.BoolOp):
        joiner = " & " if isinstance(node.op, ast.And) else " | "
        return "(" + joiner.join(_r_expr(value, space) for value in node.values) + ")"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return f"!({_r_expr(node.operand, space)})"
    if isinstance(node, ast.Subscript):
        name = _parent_name(node)
        if not isinstance(space.params[name], Boolean):
            raise ValueError("Bare condition parents must be boolean parameters.")
        return f'{name} == "TRUE"'
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.comparators[0], ast.Constant):
        name = _parent_name(node.left)
        spec = space.params[name]
        op = _R_OPS.get(type(node.ops[0]))
        value = node.comparators[0].value
        if op is None:
            raise ValueError("Supported comparisons are ==, !=, <, <=, > and >=.")
        if isinstance(spec, (Real, Int)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"Numeric parent '{name}' needs a numeric literal.")
            return f"{name} {op} {_number(value)}"
        if op not in ("==", "!="):
            raise ValueError(f"Ordering comparisons need a numeric parent, not '{name}'.")
        return f'{name} {op} "{_token_of(name, spec, value)}"'
    raise ValueError("Expected a comparison of cfg['parent'] with a scalar literal, a boolean parent, or and/or/not.")


def _r_condition(expr: str, space: ParamSpace) -> str:
    """One VAMOS condition in R syntax, with VAMOS's semantics for inactive parents.

    irace stores an inactive parent as NA and treats an NA condition as false, as VAMOS treats
    a missing parent. Under ``or`` and ``not`` R can still return TRUE (``NA | TRUE``), so those
    conditions also require every parent they read to be active.
    """
    tree = ast.parse(expr, mode="eval").body
    text = _r_expr(tree, space)
    if any(isinstance(n, (ast.BoolOp, ast.UnaryOp)) and isinstance(n.op, (ast.Or, ast.Not)) for n in ast.walk(tree)):
        body = text if isinstance(tree, ast.BoolOp) else f"({text})"
        return " & ".join([*(f"!is.na({parent})" for parent in _condition_parents(expr)), body])
    return text


def _domain(name: str, spec: ParamType) -> tuple[str, str]:
    if isinstance(spec, (Categorical, Boolean)):
        return "c", "(" + ", ".join(f'"{token}"' for token in _tokens(name, spec)) + ")"
    if isinstance(spec, (Real, Int)):
        kind = ("i" if isinstance(spec, Int) else "r") + (",log" if spec.log else "")
        return kind, f"({_number(spec.low)}, {_number(spec.high)})"
    raise TypeError(f"Unsupported param spec type for '{name}': {type(spec)!r}")


def parameter_file(space: ParamSpace) -> str:
    """Text of the irace parameter file for ``space``; untranslatable spaces fail before irace starts."""
    _dependency_order(space)
    expressions: dict[str, list[str]] = {name: [] for name in space.params}
    for condition in space.conditions:
        expressions[condition.param_name].append(condition.expr)
    lines = []
    for name, spec in space.params.items():
        if not re.fullmatch(_NAME, name):
            raise ValueError(f"irace parameter names start with a letter and use letters, digits or '_': {name!r}.")
        kind, domain = _domain(name, spec)
        conditions = []
        for expr in expressions[name]:
            try:
                conditions.append(_r_condition(expr, space))
            except (KeyError, TypeError, ValueError, SyntaxError) as exc:
                raise ValueError(f"Cannot translate condition for '{name}' to irace: {expr!r}. {exc}") from exc
        line = f'{name} "--{name}=" {kind} {domain}'
        if conditions:
            line += " | " + (conditions[0] if len(conditions) == 1 else " & ".join(f"({c})" for c in conditions))
        lines.append(line)
    return "\n".join(lines) + "\n"


def configurations_file(space: ParamSpace, configs: Sequence[Mapping[str, Any]]) -> str:
    """Text of an irace configurations file with the given active assignments."""
    names = list(space.params)
    rows = [" ".join(names)]
    for config in configs:
        active = _active_assignment(space, dict(config))
        if set(active) != set(config):
            raise ValueError(f"Initial configuration has inactive or unknown parameters: {sorted(set(config) - set(active))}.")
        space.validate(active)
        cells = []
        for name in names:
            spec = space.params[name]
            if name not in active:
                cells.append("NA")
            elif isinstance(spec, (Categorical, Boolean)):
                cells.append(_token_of(name, spec, active[name]))
            else:
                cells.append(_number(active[name]))
        rows.append(" ".join(cells))
    return "\n".join(rows) + "\n"


def decode_switches(space: ParamSpace, switches: Sequence[str]) -> dict[str, Any]:
    """Active assignment encoded by irace's ``--name=token`` arguments."""
    values: dict[str, Any] = {}
    for switch in switches:
        name, sep, token = switch[2:].partition("=")
        if not switch.startswith("--") or not sep or name not in space.params or name in values:
            raise ValueError(f"Unexpected irace argument {switch!r}.")
        spec = space.params[name]
        if isinstance(spec, Real):
            values[name] = float(token)
        elif isinstance(spec, Int):
            values[name] = int(token)
        else:
            tokens = _tokens(name, spec)
            if token not in tokens:
                raise ValueError(f"Unexpected irace value {token!r} for '{name}'.")
            values[name] = tokens[token]
    active = _active_assignment(space, values)
    if set(active) != set(values):
        raise ValueError(f"irace sent inactive parameters: {sorted(set(values) - set(active))}.")
    space.validate(active)
    return active


__all__ = ["configurations_file", "decode_switches", "parameter_file"]
