from __future__ import annotations

from dataclasses import dataclass, field

from vamos.foundation.encoding import normalize_encoding
from vamos.foundation.problem.registry.specs import ProblemSpec, get_problem_specs


@dataclass
class ProblemInfo:
    name: str
    description: str
    categories: list[str] = field(default_factory=list)
    default_n_variables: int | None = None
    default_n_objectives: int | None = None
    tags: list[str] = field(default_factory=list)
    encoding: str = "real"


def _spec_to_info(spec: ProblemSpec) -> ProblemInfo:
    # One name per encoding: "continuous" problems are "real" ones.
    encoding = normalize_encoding(spec.encoding, default="real")
    return ProblemInfo(
        name=spec.key,
        description=spec.description,
        categories=[encoding],
        default_n_variables=spec.default_n_var,
        default_n_objectives=spec.default_n_obj,
        tags=[spec.label],
        encoding=encoding,
    )


def list_problems() -> list[ProblemInfo]:
    specs = get_problem_specs()
    return [_spec_to_info(spec) for spec in specs.values()]


def get_problem_info(name: str) -> ProblemInfo | None:
    spec = get_problem_specs().get(name)
    if spec is None:
        return None
    return _spec_to_info(spec)


__all__ = ["ProblemInfo", "list_problems", "get_problem_info"]
