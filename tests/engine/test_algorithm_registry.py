from __future__ import annotations

from collections.abc import Callable
from importlib import metadata
from typing import Any

import numpy as np
import pytest

import vamos.engine.algorithm.registry as registry_mod
from vamos.engine.algorithm.config import AlgorithmConfigMapping, GenericAlgorithmConfig
from vamos.foundation.encoding import EncodingLike
from vamos.foundation.kernel.backend import KernelBackend
from vamos.foundation.problem.types import ProblemProtocol


class _FakeEntryPoint:
    def __init__(self, name: str, builder: Callable[..., Any]) -> None:
        self.name = name
        self._builder = builder

    def load(self) -> Callable[..., Any]:
        return self._builder


class _UnusedProblem:
    n_var = 1
    n_obj = 2
    n_constraints = 0
    xl: float | int | np.ndarray[Any, Any] = 0.0
    xu: float | int | np.ndarray[Any, Any] = 1.0
    encoding: EncodingLike = "real"

    def evaluate(self, X: Any, out: Any) -> None:
        raise AssertionError("The plugin fixture must not evaluate its problem.")


def test_algorithm_registry_loads_entry_point_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    class _PluginAlgorithm:
        def __init__(self, cfg: dict[str, object]) -> None:
            self.cfg = cfg

        def run(
            self,
            problem: ProblemProtocol,
            termination: tuple[str, Any] = ("max_evaluations", 1),
            seed: int = 0,
            eval_strategy: Any | None = None,
            live_viz: Any | None = None,
        ) -> dict[str, Any]:
            answer = self.cfg["answer"]
            assert isinstance(answer, (int, float))
            return {"F": np.array([[float(answer), 0.0]]), "X": None}

    def _build_plugin_algorithm(
        cfg: AlgorithmConfigMapping,
        kernel: KernelBackend | None = None,
    ) -> registry_mod.AlgorithmLike:
        return _PluginAlgorithm(dict(cfg))

    monkeypatch.setattr(registry_mod, "_ALGORITHMS", None)
    monkeypatch.setattr(registry_mod, "_ALGORITHM_PLUGINS_LOADED", False)
    monkeypatch.setattr(
        metadata,
        "entry_points",
        lambda **kwargs: (
            [_FakeEntryPoint("_pytest_plugin_algo", _build_plugin_algorithm)] if kwargs.get("group") == "vamos.algorithms" else []
        ),
    )

    loaded = registry_mod.discover_algorithm_plugins(force=True)
    registry = registry_mod.get_algorithms_registry()

    assert loaded == ("_pytest_plugin_algo",)
    algorithm = registry["_pytest_plugin_algo"](GenericAlgorithmConfig({"answer": 42}).to_dict(), None)
    result = algorithm.run(problem=_UnusedProblem())
    np.testing.assert_allclose(result["F"], np.array([[42.0, 0.0]]))


def _fresh_registry(monkeypatch: pytest.MonkeyPatch, entry_points: list[_FakeEntryPoint]) -> None:
    monkeypatch.setattr(registry_mod, "_ALGORITHMS", None)
    monkeypatch.setattr(registry_mod, "_ALGORITHM_PLUGINS_LOADED", False)
    monkeypatch.setattr(registry_mod, "_entry_points_for_group", lambda group: list(entry_points))


def _plugin_builder(cfg: AlgorithmConfigMapping, kernel: KernelBackend | None = None) -> registry_mod.AlgorithmLike:
    raise AssertionError("The plugin fixture must not be built.")


def test_plugin_named_like_builtin_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plugin with a built-in name is ignored with a warning, not silently (VR-105)."""
    _fresh_registry(monkeypatch, [_FakeEntryPoint("NSGAII", _plugin_builder)])

    with pytest.warns(RuntimeWarning, match="'NSGAII'.*built-in"):
        registry = registry_mod.get_algorithms_registry()

    assert registry["nsgaii"] is not _plugin_builder


def test_plugin_discovery_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """VAMOS_DISABLE_PLUGINS=1 keeps the stable API from loading entry points (VR-105)."""
    _fresh_registry(monkeypatch, [_FakeEntryPoint("_pytest_disabled_plugin", _plugin_builder)])
    monkeypatch.setenv("VAMOS_DISABLE_PLUGINS", "1")

    assert "_pytest_disabled_plugin" not in registry_mod.get_algorithms_registry()
    # Explicit discovery still loads them.
    assert registry_mod.discover_algorithm_plugins() == ("_pytest_disabled_plugin",)


def test_compatibility_snapshot_ignores_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    """The 1.0 snapshot lists the built-in algorithms, whatever plugins are installed (VR-105)."""
    import importlib.util
    from pathlib import Path

    _fresh_registry(monkeypatch, [_FakeEntryPoint("_pytest_snapshot_plugin", _plugin_builder)])
    path = Path(__file__).resolve().parents[2] / "tools" / "generate_v1_compatibility_snapshots.py"
    spec = importlib.util.spec_from_file_location("_v1_snapshot_generator_plugins", path)
    assert spec is not None and spec.loader is not None
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    assert generator._algorithm_snapshot()["algorithm_ids"] == list(registry_mod.get_builtin_algorithm_names())
