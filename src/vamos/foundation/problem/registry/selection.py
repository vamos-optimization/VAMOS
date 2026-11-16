from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
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
    #: The problem's other options, which ``with_options`` checks against the factory's signature.
    options: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}), hash=False)

    def instantiate(self) -> ProblemProtocol:
        """Create a new problem instance with the resolved dimensions and options."""
        return cast(ProblemProtocol, self.spec.build(self.n_var, self.n_obj, self.options))

    def with_options(self, options: Mapping[str, object]) -> ProblemSelection:
        """Return this selection with the problem's other options, such as ``capacity_ratio`` for ``bin_knapsack``.

        The options are the factory's keyword-only parameters (``spec.option_defaults()``); an option it does not
        declare, or a value other than ``None``, a boolean, a string, an integer or a finite number that a float
        represents exactly, raises ``ConfigurationError``. The selection keeps the values as a run artifact records
        them (``spec.check_options``), so replay builds the same problem.
        """
        return replace(self, options=MappingProxyType(self.spec.check_options(options)))


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
