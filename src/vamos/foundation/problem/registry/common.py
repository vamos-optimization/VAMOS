from __future__ import annotations

import inspect
import math
import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from vamos.foundation.exceptions import ConfigurationError

# A factory takes the resolved n_var and n_obj; the problem's other options are its keyword-only parameters.
ProblemFactory = Callable[[int, int | None], object]
DefaultNVarFn = Callable[[int], int]


@dataclass(frozen=True)
class ProblemSpec:
    """Metadata and factory for a benchmark problem."""

    key: str
    label: str
    default_n_var: int
    default_n_obj: int
    allow_n_obj_override: bool
    factory: ProblemFactory
    default_n_var_fn: DefaultNVarFn | None = None
    description: str = ""
    encoding: str = "real"
    # The factory always builds ``default_n_var`` variables, so another ``n_var`` is rejected.
    fixed_n_var: bool = False

    def resolve_dimensions(self, *, n_var: int | None, n_obj: int | None) -> tuple[int, int]:
        """
        Apply default dimensions and enforce override rules.

        A problem with a fixed number of objectives or variables rejects a different
        ``n_obj`` or ``n_var`` with ``ValueError``, so the selection always reports the
        dimensions of the instance it creates. A dimension that is not an integer
        (a float, a string or a boolean) is rejected with ``ValueError``.
        """
        n_var = _optional_dimension("n_var", n_var)
        n_obj = _optional_dimension("n_obj", n_obj)
        if self.allow_n_obj_override:
            actual_n_obj = n_obj if n_obj is not None else self.default_n_obj
            if actual_n_obj <= 0:
                raise ValueError("n_obj must be a positive integer.")
        else:
            actual_n_obj = self.default_n_obj
            if n_obj is not None and n_obj != actual_n_obj:
                raise ValueError(
                    f"Problem '{self.label}' has a fixed number of objectives ({self.default_n_obj}). --n-obj overrides are not supported."
                )

        if self.fixed_n_var and n_var is not None and n_var != self.default_n_var:
            raise ValueError(
                f"Problem '{self.label}' has a fixed number of variables ({self.default_n_var}). --n-var overrides are not supported."
            )
        if n_var is None:
            actual_n_var = self.default_n_var_fn(actual_n_obj) if self.default_n_var_fn is not None else self.default_n_var
        else:
            actual_n_var = n_var
        if actual_n_var <= 0:
            raise ValueError("n_var must be a positive integer.")

        return actual_n_var, actual_n_obj

    def option_defaults(self) -> dict[str, object]:
        """The problem's options besides its dimensions, with their defaults: the factory's keyword-only parameters."""
        parameters = inspect.signature(self.factory).parameters.values()
        return {parameter.name: parameter.default for parameter in parameters if parameter.kind is inspect.Parameter.KEYWORD_ONLY}

    def check_options(self, options: Mapping[str, object]) -> None:
        """Reject an option the factory does not declare, and a value that a run artifact cannot record exactly.

        A value is ``None``, a boolean, a string, an integer or a finite number, so that the resolved spec
        records it as JSON and replay passes the same value to the factory.
        """
        dimensions = [name for name in ("n_var", "n_obj") if name in options]
        if dimensions:
            raise ConfigurationError(
                f"{' and '.join(dimensions)} of problem '{self.key}' are dimensions, not options.",
                suggestion="Pass n_var and n_obj to make_problem_selection.",
            )
        declared = self.option_defaults()
        unknown = sorted(str(name) for name in options if name not in declared)
        if unknown:
            names = ["n_var", "n_obj", *declared]
            accepted = f"{', '.join(names[:-1])} and {names[-1]}" if declared else "no options besides n_var and n_obj"
            raise ConfigurationError(
                f"Problem '{self.key}' accepts {accepted}; got {', '.join(unknown)}.",
                suggestion="Build the problem instance yourself to pass other arguments.",
                details={"problem": self.key, "unknown": unknown, "accepted": list(declared)},
            )
        for name, value in options.items():
            if not _is_recordable(value):
                raise ConfigurationError(
                    f"Option {name}={value!r} of problem '{self.key}' must be None, a boolean, a string, an integer or a "
                    "finite number, so that the run artifact records it and replay rebuilds the same problem.",
                    details={"problem": self.key, "option": str(name), "type": type(value).__name__},
                )

    def build(self, n_var: int, n_obj: int, options: Mapping[str, object]) -> object:
        """Call the factory with the dimensions and the options."""
        factory: Callable[..., object] = self.factory
        return factory(n_var, n_obj, **options)


def _is_recordable(value: object) -> bool:
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, numbers.Integral):
        return True
    return isinstance(value, numbers.Real) and math.isfinite(float(value))


def _optional_dimension(name: str, value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer; got {value!r}.")
    return int(value)


__all__ = ["ProblemSpec", "ProblemFactory"]
