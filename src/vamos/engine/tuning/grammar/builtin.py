"""The built-in parameter spaces, packaged with VAMOS as YAML grammars.

``vamos/resources/parameter_spaces/<space>.yaml`` holds one grammar per built-in tuning space and is
the source of truth: the grammars were written once from the Python builders of VAMOS 1.3.0 and are
maintained as YAML. VAMOS does not depend on PyYAML, so at run time it reads
``parameter_spaces.json``, the canonical form of the same grammars that
``python tools/update_parameter_spaces.py`` writes from the YAML files; a test fails when the two
differ.
"""

from __future__ import annotations

import json
from functools import cache, lru_cache
from importlib import resources
from typing import Any

from .document import parameter_space_from_document
from .model import SCHEMA, SCHEMA_VERSION, ParameterSpace

BUNDLE = "parameter_spaces.json"


@lru_cache(maxsize=1)
def _bundle() -> dict[str, Any]:
    text = (resources.files("vamos.resources") / "parameter_spaces" / BUNDLE).read_text(encoding="utf-8")
    document = json.loads(text)
    if document.get("schema") != SCHEMA or document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{BUNDLE} holds {document.get('schema')} {document.get('schema_version')}, not {SCHEMA} {SCHEMA_VERSION}.")
    spaces: dict[str, Any] = document["spaces"]
    return spaces


def builtin_parameter_space_names() -> tuple[str, ...]:
    """The names of the built-in spaces, such as ``nsgaii`` or ``moead_permutation``."""
    return tuple(_bundle())


@cache
def builtin_parameter_space(name: str) -> ParameterSpace:
    """The built-in space ``name``, read and checked against rules G1 to G6."""
    spaces = _bundle()
    if name not in spaces:
        raise KeyError(f"No built-in parameter space {name!r}; the spaces are {', '.join(spaces)}.")
    return parameter_space_from_document(spaces[name])


__all__ = ["builtin_parameter_space", "builtin_parameter_space_names"]
