"""Registered problems and built-in algorithms as the Studio catalog lists them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.problem.registry import get_problem_specs
from vamos.foundation.problem.resolver import resolve_reference_front_path
from vamos.ux.studio.algorithm_catalog import (
    algorithm_label,
    algorithms_for_encoding,
    config_space_for,
    population_is_derived,
    studio_algorithms,
)
from vamos.ux.studio.algorithm_form import VAMOS_DEFAULT_CHOICE, FormParameter, build_algorithm_form, choice_label, format_config_value

ENCODINGS = ("real", "binary", "integer", "permutation", "mixed")


@dataclass(frozen=True)
class ProblemEntry:
    """A registered problem with its default dimensions."""

    key: str
    label: str
    description: str
    encoding: str
    n_var: int
    n_obj: int
    n_obj_adjustable: bool
    reference_front: bool
    algorithms: tuple[str, ...]


@dataclass(frozen=True)
class AlgorithmEntry:
    """A built-in algorithm and the problem encodings it has operators for."""

    key: str
    label: str
    encodings: tuple[str, ...]


def problem_entries() -> list[ProblemEntry]:
    """Every registered problem, in registry order."""
    algorithms = {encoding: algorithms_for_encoding(encoding) for encoding in ENCODINGS}
    entries: list[ProblemEntry] = []
    for key, spec in get_problem_specs().items():
        n_var, n_obj = spec.resolve_dimensions(n_var=None, n_obj=None)
        encoding = normalize_encoding(spec.encoding)
        entries.append(
            ProblemEntry(
                key=key,
                label=spec.label,
                description=spec.description,
                encoding=encoding,
                n_var=n_var,
                n_obj=n_obj,
                n_obj_adjustable=spec.allow_n_obj_override,
                reference_front=has_reference_front(key, n_obj),
                algorithms=algorithms.get(encoding, ()),
            )
        )
    return entries


def has_reference_front(problem: str, n_obj: int) -> bool:
    """Whether VAMOS ships a reference front of ``problem`` with ``n_obj`` objectives."""
    path = resolve_reference_front_path(problem, None, n_obj=n_obj)
    if path is None:
        return False
    try:
        with Path(path).open(encoding="utf-8") as handle:
            first = next((line for line in handle if line.strip()), "")
    except OSError:
        return False
    return len(first.split(",")) == n_obj


def problem_rows(entries: list[ProblemEntry]) -> list[dict[str, object]]:
    return [
        {
            "Problem": entry.label,
            "Key": entry.key,
            "Encoding": entry.encoding,
            "Variables": entry.n_var,
            "Objectives": f"{entry.n_obj} (adjustable)" if entry.n_obj_adjustable else str(entry.n_obj),
            "Reference front": "yes" if entry.reference_front else "no",
            "Algorithms": len(entry.algorithms),
        }
        for entry in entries
    ]


def filter_problems(entries: list[ProblemEntry], *, encoding: str | None = None, text: str = "") -> list[ProblemEntry]:
    """Problems of ``encoding`` whose key, label or description contains ``text``."""
    needle = text.strip().lower()
    return [
        entry
        for entry in entries
        if (encoding is None or entry.encoding == encoding)
        and (not needle or needle in f"{entry.key} {entry.label} {entry.description}".lower())
    ]


def algorithm_entries() -> list[AlgorithmEntry]:
    """Every built-in algorithm, with the encodings that have a configuration space."""
    return [
        AlgorithmEntry(
            key=name,
            label=algorithm_label(name),
            encodings=tuple(encoding for encoding in ENCODINGS if config_space_for(name, encoding) is not None),
        )
        for name in studio_algorithms()
    ]


def representative_problem(encoding: str, entries: list[ProblemEntry] | None = None) -> ProblemEntry | None:
    """The first registered problem of ``encoding``, whose dimensions the algorithm details use."""
    return next((entry for entry in entries or problem_entries() if entry.encoding == encoding), None)


def population_rule(algorithm: str, n_obj: int) -> str:
    if algorithm in {"nsgaiii", "rvea"}:
        return "The population size equals the number of reference directions for the number of objectives."
    if population_is_derived(algorithm, n_obj):
        return "With three or more objectives, the population size equals the number of weight vectors."
    return "The population size is a free setting."


def parameter_rows(algorithm: str, problem: ProblemEntry) -> list[dict[str, object]]:
    """The configurable parameters of ``algorithm`` for ``problem``: default value and allowed values."""
    form = build_algorithm_form(algorithm, encoding=problem.encoding, n_var=problem.n_var, n_obj=problem.n_obj, pop_size=100)
    if form is None:
        return []
    return [
        {
            "Parameter": ("  " * (parameter.depth - 1) + "↳ " if parameter.depth else "") + parameter.label,
            "VAMOS default": _default_text(parameter),
            "Allowed values": _allowed_values(parameter),
        }
        for parameter in form.parameters
    ]


def _default_text(parameter: FormParameter) -> str:
    if parameter.kind == "boolean":
        return "yes" if parameter.start else "no"
    if parameter.kind == "categorical":
        return choice_label(parameter, parameter.start)
    if not parameter.from_default:
        return "none (set when customized)"
    return format_config_value(parameter.start)


def _allowed_values(parameter: FormParameter) -> str:
    if parameter.kind == "boolean":
        return "yes / no"
    if parameter.kind == "categorical":
        return ", ".join(format_config_value(choice) for choice in parameter.choices if choice != VAMOS_DEFAULT_CHOICE)
    scale = " (log scale)" if parameter.log else ""
    return f"{format_config_value(parameter.low)} to {format_config_value(parameter.high)}{scale}"


__all__ = [
    "ENCODINGS",
    "AlgorithmEntry",
    "ProblemEntry",
    "algorithm_entries",
    "filter_problems",
    "has_reference_front",
    "parameter_rows",
    "population_rule",
    "problem_entries",
    "problem_rows",
    "representative_problem",
]
