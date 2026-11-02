from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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
        dimensions of the instance it creates.
        """
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


__all__ = ["ProblemSpec", "ProblemFactory"]
