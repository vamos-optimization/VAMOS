"""The parameter-space grammar: a strict tree of parameters, each with a default.

A categorical parameter chooses between productions, one per value; the production of a value holds
the parameters that the value activates, and the parameter's global sub-parameters are active whatever
its value. Numeric and boolean parameters are terminals. Every parameter has exactly one parent, so a
configuration is a derivation of the tree: listing its active parameters in depth-first order
determines it. See ``docs/dev/vamos-config-design.md``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from vamos.engine.algorithm.variants import ALGORITHM_VARIANT_GROUPS

SCHEMA = "vamos.parameter-space"
SCHEMA_VERSION = "1.0.0"
CONFIGURATION_SCHEMA = "vamos.parameter-configuration"
CONFIGURATION_SCHEMA_VERSION = "1.0.0"

ParameterType = Literal["categorical", "real", "int", "boolean"]
Scalar = str | int | float | bool


class ParameterSpaceError(ValueError):
    """A space or a configuration that breaks a rule of the ``vamos.parameter-space`` schema.

    ``rule`` is the rule it breaks (``"G1"`` to ``"G7"``, or ``"structure"`` for a document that is not
    shaped as the schema says), ``path`` where, as dotted names from the top of the document.
    """

    def __init__(self, rule: str, path: str, message: str) -> None:
        self.rule = rule
        self.path = path
        where = f" at {path}" if path else ""
        super().__init__(f"{rule}{where}: {message}")


def same_scalar(a: object, b: object) -> bool:
    """Equality that keeps booleans apart from numbers (``True`` is not ``1``) and integers from floats."""
    return type(a) is type(b) and a == b if isinstance(a, bool) or isinstance(b, bool) else a == b


@dataclass(frozen=True)
class Parameter:
    """One node of the grammar.

    A categorical parameter has ``values`` and, aligned with them, ``productions``: the parameters that
    each value activates (an empty tuple for a value that activates none). A real or integer parameter
    has ``range``, both ends included, and ``log`` for a log-uniform scale.
    """

    name: str
    type: ParameterType
    default: Scalar
    range: tuple[float, float] | None = None
    log: bool = False
    values: tuple[Scalar, ...] = ()
    productions: tuple[tuple[Parameter, ...], ...] = ()
    global_sub_parameters: tuple[Parameter, ...] = ()
    role: str | None = None
    label: str | None = None
    description: str | None = None

    def production(self, value: object) -> tuple[Parameter, ...]:
        """The parameters that ``value`` activates; empty for a terminal or a value without a production."""
        for candidate, production in zip(self.values, self.productions):
            if same_scalar(candidate, value):
                return production
        return ()

    def children(self, value: object) -> tuple[Parameter, ...]:
        """The parameters active under this one when it takes ``value``: its global sub-parameters, then the production."""
        return (*self.global_sub_parameters, *self.production(value))

    def contains(self, value: object) -> bool:
        """Whether ``value`` belongs to the parameter's domain."""
        if self.type == "categorical":
            return any(same_scalar(candidate, value) for candidate in self.values)
        if self.type == "boolean":
            return isinstance(value, bool)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or self.range is None:
            return False
        if self.type == "int" and not isinstance(value, int):
            return False
        low, high = self.range
        return math.isfinite(float(value)) and low <= value <= high

    def sample(self, rng: np.random.Generator) -> Scalar:
        """A value drawn from the domain: uniformly, or log-uniformly with ``log``."""
        if self.type == "categorical":
            return self.values[int(rng.integers(0, len(self.values)))]
        if self.type == "boolean":
            return bool(rng.integers(0, 2))
        assert self.range is not None
        low, high = self.range
        if self.type == "int":
            if self.log:
                drawn = int(round(math.exp(rng.uniform(math.log(low), math.log(high)))))
                return int(min(max(drawn, int(low)), int(high)))
            return int(rng.integers(int(low), int(high) + 1))
        if self.log:
            return float(min(max(math.exp(rng.uniform(math.log(low), math.log(high))), low), high))
        return float(rng.uniform(low, high))

    def descendants(self) -> Iterator[Parameter]:
        """Every parameter below this one, under every value, in depth-first order."""
        for child in self.global_sub_parameters:
            yield child
            yield from child.descendants()
        for production in self.productions:
            for child in production:
                yield child
                yield from child.descendants()


@dataclass(frozen=True)
class SpaceIdentity:
    """How configurations and run manifests name a space: its name, schema version and SHA-256."""

    name: str
    schema_version: str
    sha256: str

    def to_document(self) -> dict[str, str]:
        return {"name": self.name, "schema_version": self.schema_version, "sha256": self.sha256}


@dataclass(frozen=True)
class ParameterSpace:
    """A validated ``vamos.parameter-space`` document: the parameters that hang from the root, in order."""

    name: str
    algorithm: str
    encoding: str
    parameters: tuple[Parameter, ...]
    description: str | None = None
    schema_version: str = SCHEMA_VERSION
    _by_name: dict[str, Parameter] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._by_name.update({parameter.name: parameter for parameter in self.walk()})

    @property
    def variant(self) -> str:
        """The name under which ``config_from_assignment`` builds configurations of this space.

        The space's own name when VAMOS knows it as a variant of the algorithm (``nsgaii_constrained``),
        otherwise the algorithm for real-coded spaces and ``<algorithm>_<encoding>`` for the others.
        """
        if self.name in ALGORITHM_VARIANT_GROUPS.get(self.algorithm, set()):
            return self.name
        return self.algorithm if self.encoding == "real" else f"{self.algorithm}_{self.encoding}"

    def walk(self) -> Iterator[Parameter]:
        """Every parameter of the space, under every value, in depth-first order."""
        for parameter in self.parameters:
            yield parameter
            yield from parameter.descendants()

    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    def parameter(self, name: str) -> Parameter:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(f"The space {self.name!r} has no parameter {name!r}.") from None

    def active_parameters(self, values: Mapping[str, object]) -> tuple[Parameter, ...]:
        """The parameters active in the derivation that ``values`` chooses, in depth-first order.

        A categorical parameter missing from ``values`` contributes no children.
        """
        active: list[Parameter] = []

        def visit(parameters: tuple[Parameter, ...]) -> None:
            for parameter in parameters:
                active.append(parameter)
                if parameter.type == "categorical" and parameter.name in values:
                    visit(parameter.children(values[parameter.name]))

        visit(self.parameters)
        return tuple(active)

    @property
    def identity(self) -> SpaceIdentity:
        """The space's name, schema version and the SHA-256 of its canonical JSON form (see :mod:`.document`)."""
        from .document import space_sha256

        return SpaceIdentity(self.name, self.schema_version, space_sha256(self))

    def default_values(self) -> dict[str, Scalar]:
        """The default configuration: every parameter that the defaults activate, at its default."""
        return self._derive(lambda parameter: parameter.default)

    def sample(self, rng: np.random.Generator) -> dict[str, Scalar]:
        """A random configuration: a production at every categorical node, every active terminal drawn.

        Only the active parameters draw values, in depth-first order.
        """
        return self._derive(lambda parameter: parameter.sample(rng))

    def _derive(self, choose: Callable[[Parameter], Scalar]) -> dict[str, Scalar]:
        values: dict[str, Scalar] = {}

        def visit(parameters: tuple[Parameter, ...]) -> None:
            for parameter in parameters:
                value = choose(parameter)
                values[parameter.name] = value
                visit(parameter.children(value))

        visit(self.parameters)
        return values


__all__ = [
    "CONFIGURATION_SCHEMA",
    "CONFIGURATION_SCHEMA_VERSION",
    "Parameter",
    "ParameterSpace",
    "ParameterSpaceError",
    "ParameterType",
    "SCHEMA",
    "SCHEMA_VERSION",
    "Scalar",
    "SpaceIdentity",
    "same_scalar",
]
