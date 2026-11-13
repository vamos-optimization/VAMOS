from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from vamos.foundation.exceptions import InvalidProblemError

from ..types import ProblemProtocol
from .specs import ProblemSpec, get_problem_specs


@dataclass(frozen=True)
class ProblemSelection:
    """Concrete problem instance choice with resolved dimensions."""

    spec: ProblemSpec
    n_var: int
    n_obj: int

    def instantiate(self) -> ProblemProtocol:
        """Create a new problem instance with the resolved dimensions."""
        return cast(ProblemProtocol, self.spec.factory(self.n_var, self.n_obj))


def make_problem_selection(key: str, *, n_var: int | None = None, n_obj: int | None = None) -> ProblemSelection:
    """Look up a registered problem by *key* and resolve its dimensions.

    Parameters
    ----------
    key : str
        Registered problem name (e.g. ``"zdt1"``, ``"dtlz2"``).
    n_var : int, optional
        Override the default number of decision variables.
    n_obj : int, optional
        Override the default number of objectives (only for problems
        that allow it).

    Returns
    -------
    ProblemSelection
        A frozen selection with the resolved ``spec``, ``n_var``, and
        ``n_obj`` ready to instantiate.

    Raises
    ------
    InvalidProblemError
        If *key* does not match any registered problem. It is also a
        ``KeyError``; its message suggests close matches and its
        ``details["available"]`` lists every registered name.
    """
    specs = get_problem_specs()
    try:
        spec = specs[key]
    except KeyError as exc:
        available = sorted(specs.keys())
        raise InvalidProblemError(key, available) from exc

    actual_n_var, actual_n_obj = spec.resolve_dimensions(n_var=n_var, n_obj=n_obj)
    return ProblemSelection(spec=spec, n_var=actual_n_var, n_obj=actual_n_obj)


__all__ = ["ProblemSelection", "make_problem_selection"]
