import importlib

import pytest


def test_import_vamos_minimal():
    pkg = importlib.import_module("vamos")
    api = importlib.import_module("vamos.api")
    assert hasattr(api, "optimize")
    assert pkg is not None


def test_kernel_registry_minimal(monkeypatch):
    registry = importlib.import_module("vamos.foundation.kernel.registry")

    # NumPy backend should always be available.
    kernel = registry.resolve_kernel("numpy")
    assert kernel is not None

    # Simulate missing optional backends to verify clear error messages.
    real_import = registry.import_module

    def fake_import(name, *args, **kwargs):
        if name in ("vamos.foundation.kernel.numba_backend", "vamos.foundation.kernel.moocore_backend"):
            raise ImportError("forced-missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(registry, "import_module", fake_import)

    with pytest.raises(ImportError, match="numba"):
        registry.resolve_kernel("numba")
    with pytest.raises(ImportError, match="moocore"):
        registry.resolve_kernel("moocore")
