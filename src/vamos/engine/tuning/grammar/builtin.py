"""The built-in parameter spaces, packaged with VAMOS as YAML grammars.

``vamos/resources/parameter_spaces/<space>.yaml`` holds one grammar per built-in tuning space and is
the source of truth, which VAMOS reads at run time: the grammars were written once from the Python
builders of VAMOS 1.3.0 and are maintained as YAML. ``python tools/update_parameter_spaces.py``
checks every file and writes the test snapshot of what the bridges build from them.
"""

from __future__ import annotations

from functools import cache, lru_cache
from importlib import resources
from typing import Any

from .loading import parameter_space_from_yaml
from .model import ParameterSpace

SUFFIX = ".yaml"


@lru_cache(maxsize=1)
def _grammar_files() -> dict[str, Any]:
    """The packaged grammar files by space name, in name order."""
    folder = resources.files("vamos.resources") / "parameter_spaces"
    files = sorted((entry for entry in folder.iterdir() if entry.name.endswith(SUFFIX)), key=lambda entry: entry.name)
    return {entry.name[: -len(SUFFIX)]: entry for entry in files}


def builtin_parameter_space_names() -> tuple[str, ...]:
    """The names of the built-in spaces, such as ``nsgaii`` or ``moead_permutation``."""
    return tuple(_grammar_files())


@cache
def builtin_parameter_space(name: str) -> ParameterSpace:
    """The built-in space ``name``, read from its YAML file and checked against rules G1 to G6."""
    files = _grammar_files()
    if name not in files:
        raise KeyError(f"No built-in parameter space {name!r}; the spaces are {', '.join(files)}.")
    space = parameter_space_from_yaml(files[name].read_text(encoding="utf-8"))
    if space.name != name:
        raise ValueError(f"{name}{SUFFIX} holds the space {space.name!r}: a grammar file is named after its space.")
    return space


__all__ = ["builtin_parameter_space", "builtin_parameter_space_names"]
