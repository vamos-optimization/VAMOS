"""Parameter spaces as strict grammars with defaults: the ``vamos.parameter-space`` 1.0.0 schema.

A space is a tree of parameters in the vocabulary of the algorithm configurations; a configuration
is a derivation of it. Rules G1 to G6 are checked whenever a space is read
(:func:`parameter_space_from_document`, :func:`load_parameter_space`); rule G7 runs the algorithm
and is checked on demand (:func:`check_default_configuration`).

| Rule | Statement |
| --- | --- |
| G1 | Parameter names are unique in the space and match ``^[a-z][a-z0-9_]*$``. |
| G2 | Each parameter has one parent; there are no expressions. |
| G3 | Every ``default`` belongs to its domain. |
| G4 | Ranges are ordered, ``log`` needs a positive lower end, and integer ranges have integer ends. |
| G5 | ``values`` is not empty and has no duplicates; a value with a production is a string. |
| G6 | ``schema`` and ``schema_version`` are known; ``algorithm`` and ``encoding`` exist in VAMOS. |
| G7 | The default configuration runs as the algorithm's ``default()`` configuration. |

The schema has no forbidden clauses, because a space offers only configurations that run, and one
range per parameter, the same for tuning and for editing by hand. The built-in tuning spaces are
grammars packaged with VAMOS (:func:`builtin_parameter_space`), and the tuners sample their flat
views (:func:`to_algorithm_config_space`). This package is Experimental, as the tuning spaces are.
"""

from __future__ import annotations

from .builtin import builtin_parameter_space, builtin_parameter_space_names
from .configuration import (
    ParameterConfiguration,
    configuration,
    configuration_from_command_line,
    configuration_from_document,
    default_configuration,
)
from .default_check import check_default_configuration, default_assignment
from .document import canonical_json, parameter_space_document, parameter_space_from_document, space_sha256
from .loading import load_parameter_space, parameter_space_from_json, parameter_space_from_yaml
from .model import (
    CONFIGURATION_SCHEMA,
    CONFIGURATION_SCHEMA_VERSION,
    SCHEMA,
    SCHEMA_VERSION,
    Parameter,
    ParameterSpace,
    ParameterSpaceError,
    SpaceIdentity,
)
from .view import to_algorithm_config_space

__all__ = [
    "CONFIGURATION_SCHEMA",
    "CONFIGURATION_SCHEMA_VERSION",
    "Parameter",
    "ParameterConfiguration",
    "ParameterSpace",
    "ParameterSpaceError",
    "SCHEMA",
    "SCHEMA_VERSION",
    "SpaceIdentity",
    "builtin_parameter_space",
    "builtin_parameter_space_names",
    "canonical_json",
    "check_default_configuration",
    "configuration",
    "configuration_from_command_line",
    "configuration_from_document",
    "default_assignment",
    "default_configuration",
    "load_parameter_space",
    "parameter_space_document",
    "parameter_space_from_document",
    "parameter_space_from_json",
    "parameter_space_from_yaml",
    "space_sha256",
    "to_algorithm_config_space",
]
