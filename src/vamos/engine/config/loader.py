"""
Config loading utilities shared by CLI and programmatic entrypoints.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any, cast

_YAML_MERGE_TAG = "tag:yaml.org,2002:merge"


def load_experiment_spec(path: str) -> dict[str, Any]:
    """
    Load a YAML or JSON experiment specification.

    A key repeated in the same mapping raises ``ValueError``: YAML and JSON
    parsers otherwise keep the last value without warning. YAML merge keys
    (``<<: *anchor``) may still be overridden by the mapping's own keys.
    """
    spec_path = Path(path).expanduser().resolve()
    if not spec_path.exists():
        raise FileNotFoundError(f"Config file '{spec_path}' does not exist.")
    suffix = spec_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("YAML config requested but PyYAML is not installed. Install with 'pip install pyyaml'.") from exc
        with spec_path.open("r", encoding="utf-8") as fh:
            # The loader is a yaml.SafeLoader subclass: it builds plain data only.
            return cast(dict[str, Any], yaml.load(fh, Loader=_unique_key_yaml_loader(yaml, spec_path)) or {})
    with spec_path.open("r", encoding="utf-8") as fh:
        return cast(dict[str, Any], json.load(fh, object_pairs_hook=lambda pairs: _unique_json_object(pairs, spec_path)))


def _unique_json_object(pairs: Iterable[tuple[str, Any]], spec_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"Duplicate key {key!r} in config file '{spec_path}'.")
        out[key] = value
    return out


def _unique_key_yaml_loader(yaml: ModuleType, spec_path: Path) -> type:
    """A ``yaml.SafeLoader`` subclass that rejects a key repeated in one mapping."""

    def construct_mapping(loader: Any, node: Any) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _value_node in node.value:
            if key_node.tag == _YAML_MERGE_TAG:
                continue
            key = loader.construct_object(key_node, deep=True)
            try:
                duplicate = key in seen
            except TypeError:  # unhashable key: the constructor reports it
                continue
            if duplicate:
                line = key_node.start_mark.line + 1
                raise ValueError(f"Duplicate key {key!r} in config file '{spec_path}' (line {line}).")
            seen.add(key)
        return cast(dict[Any, Any], loader.construct_mapping(node, deep=True))

    loader = type("_UniqueKeySafeLoader", (yaml.SafeLoader,), {})
    yaml.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping, Loader=loader)
    return loader
