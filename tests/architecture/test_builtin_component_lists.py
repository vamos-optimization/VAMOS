"""The built-in algorithms and kernel backends are listed in several places, which must agree.

``docs/dev/add_algorithm.md`` and ``docs/dev/add_backend.md`` name the same places.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path
from typing import get_args

ROOT = Path(__file__).resolve().parents[2]


def _factory_builder_names() -> set[str]:
    """The algorithms ``engine.algorithm.factory.build_algorithm`` builds without a configuration."""
    from vamos.engine.algorithm import factory

    tree = ast.parse(inspect.getsource(factory.build_algorithm))
    builders = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "builders"
    )
    assert isinstance(builders, ast.Dict)
    return {key.value for key in builders.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}


def _algorithm_lists() -> dict[str, set[str]]:
    from vamos.engine.algorithm import catalog, registry
    from vamos.engine.algorithm.config import defaults
    from vamos.experiment._execution_support import VariationConfigs
    from vamos.experiment.artifacts import component_support
    from vamos.foundation.registry import Registry

    registered: Registry[object] = Registry("built-in algorithms")
    registry._register_algorithms(registered)  # Without the installed plugins.
    return {
        "engine.algorithm.catalog.ENABLED_ALGORITHMS": set(catalog.ENABLED_ALGORITHMS),
        "engine.algorithm.catalog.ALGORITHM_MINIMUMS": set(catalog.ALGORITHM_MINIMUMS),
        "engine.algorithm.catalog._LABELS": set(catalog._LABELS),
        "engine.algorithm.registry._register_algorithms": set(registered.keys()),
        "engine.algorithm.registry.get_builtin_algorithm_names": set(registry.get_builtin_algorithm_names()),
        "engine.algorithm.config.defaults._DEFAULT_FACTORIES": set(defaults._DEFAULT_FACTORIES),
        "engine.algorithm.factory.build_algorithm": _factory_builder_names(),
        "experiment._execution_support.VariationConfigs": {field.name for field in dataclasses.fields(VariationConfigs)},
        "experiment.artifacts.component_support.BUILT_IN_ALGORITHMS": set(component_support.BUILT_IN_ALGORITHMS),
    }


def _backend_lists() -> dict[str, set[str]]:
    from vamos.engine.algorithm.config.types import EngineName
    from vamos.experiment.artifacts import component_support
    from vamos.foundation.exceptions import InvalidEngineError
    from vamos.foundation.kernel.registry import KERNELS

    return {
        "foundation.kernel.registry.KERNELS": set(KERNELS),
        "engine.algorithm.config.types.EngineName": set(get_args(EngineName)),
        "experiment.artifacts.component_support.BUILT_IN_BACKENDS": set(component_support.BUILT_IN_BACKENDS),
        "foundation.exceptions.InvalidEngineError": set(InvalidEngineError("unknown").details["available"]),
    }


def _disagreements(lists: dict[str, set[str]]) -> list[str]:
    reference_name, reference = next(iter(lists.items()))
    return [
        f"{name}: missing {sorted(reference - names)}, extra {sorted(names - reference)} (against {reference_name})"
        for name, names in lists.items()
        if names != reference
    ]


def test_builtin_algorithm_and_backend_lists_agree() -> None:
    assert not _disagreements(_algorithm_lists())
    assert not _disagreements(_backend_lists())


def test_optional_backends_get_the_compute_install_hint() -> None:
    from vamos.foundation.exceptions import BackendNotAvailableError
    from vamos.foundation.kernel.registry import KERNELS

    for backend in set(KERNELS) - {"numpy"}:
        assert "vamos-optimization[compute]" in str(BackendNotAvailableError(backend))


def test_extension_guides_name_every_list() -> None:
    algorithm_guide = (ROOT / "docs" / "dev" / "add_algorithm.md").read_text(encoding="utf-8")
    backend_guide = (ROOT / "docs" / "dev" / "add_backend.md").read_text(encoding="utf-8")

    for name in _algorithm_lists():
        module, _, symbol = name.rpartition(".")
        assert f"src/vamos/{module.replace('.', '/')}.py" in algorithm_guide and symbol in algorithm_guide, name
    for name in _backend_lists():
        module, _, symbol = name.rpartition(".")
        assert f"src/vamos/{module.replace('.', '/')}.py" in backend_guide and symbol in backend_guide, name
