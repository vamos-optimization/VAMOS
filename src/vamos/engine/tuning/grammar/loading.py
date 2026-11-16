"""Loading parameter spaces from JSON or YAML files.

YAML is read with PyYAML's safe loader (PyYAML is a dependency of VAMOS), JSON with the standard library.
Both readers reject a key repeated in one mapping, which JSON and YAML parsers otherwise resolve
silently by keeping the last value.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .document import parameter_space_from_document
from .model import ParameterSpace, ParameterSpaceError


def load_parameter_space(path: str | Path) -> ParameterSpace:
    """The space in a ``vamos.parameter-space`` file (``.json``, ``.yaml`` or ``.yml``), validated by rules G1 to G6."""
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() in {".yaml", ".yml"}:
        return parameter_space_from_yaml(text)
    if source.suffix.lower() == ".json":
        return parameter_space_from_json(text)
    raise ValueError(f"{source.name}: a parameter space is a .json, .yaml or .yml file.")


def parameter_space_from_json(text: str) -> ParameterSpace:
    """The space in a JSON ``vamos.parameter-space`` document."""
    return parameter_space_from_document(json.loads(text, object_pairs_hook=_unique_object))


def parameter_space_from_yaml(text: str) -> ParameterSpace:
    """The space in a YAML ``vamos.parameter-space`` document."""
    return parameter_space_from_document(read_yaml(text))


def read_yaml(text: str) -> Any:
    """Plain data from YAML text, rejecting repeated keys."""
    import yaml  # type: ignore[import-untyped]

    def construct_mapping(loader: Any, node: Any) -> dict[Any, Any]:
        keys = [loader.construct_object(key_node, deep=True) for key_node, _ in node.value]
        for index, key in enumerate(keys):
            if key in keys[:index]:
                raise ParameterSpaceError(
                    "G1", str(key), f"the key is repeated in one mapping (line {node.value[index][0].start_mark.line + 1})"
                )
        return dict(loader.construct_mapping(node, deep=True))

    loader: Any = type("_ParameterSpaceLoader", (yaml.SafeLoader,), {})
    loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)
    return yaml.load(text, Loader=loader)  # a SafeLoader subclass: it builds plain data only


def _unique_object(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ParameterSpaceError("G1", key, "the key is repeated in one mapping")
        result[key] = value
    return result


__all__ = ["load_parameter_space", "parameter_space_from_json", "parameter_space_from_yaml", "read_yaml"]
