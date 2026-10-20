from __future__ import annotations

from types import SimpleNamespace

import pytest

import vamos
from vamos.engine.algorithm.registry import get_builtin_algorithm_names
from vamos.experiment.runtime import catalog


@pytest.mark.parametrize("algorithm", get_builtin_algorithm_names())
@pytest.mark.parametrize("available", [False, True])
def test_auto_selection_for_every_builtin(algorithm, available, monkeypatch):
    monkeypatch.setattr(catalog, "_has_numba", lambda: available)
    assert catalog.resolve_engine_details("AUTO", algorithm=algorithm.upper()) == ("numba" if available else "numpy", "auto")


@pytest.mark.parametrize("engine,source", [(None, "default"), ("numpy", "explicit"), ("numba", "explicit"), ("moocore", "explicit")])
def test_named_and_default_engines_never_probe_or_substitute(engine, source, monkeypatch):
    def forbidden():
        raise AssertionError("availability must not change a pinned/default engine")

    monkeypatch.setattr(catalog, "_has_numba", forbidden)
    assert catalog.resolve_engine_details(engine, algorithm="nsgaii") == (engine or "numpy", source)


@pytest.mark.parametrize("algorithm", [None, "custom_algorithm", "pymoo_nsga2", "unknown"])
def test_auto_does_not_claim_plugin_compatibility(algorithm, monkeypatch):
    monkeypatch.setattr(catalog, "_has_numba", lambda: pytest.fail("must not probe optional backends"))
    assert catalog.resolve_engine_details("auto", algorithm=algorithm) == ("numpy", "auto")


@pytest.mark.parametrize("error", [ImportError, OSError, RuntimeError, AttributeError])
@pytest.mark.parametrize("stage", ["dependency", "backend"])
def test_auto_handles_unloadable_numba(error, stage, monkeypatch, caplog):
    def fail(*args):
        raise error("broken optional installation")

    monkeypatch.setattr(catalog, "import_module", lambda _: SimpleNamespace(config=SimpleNamespace(DISABLE_JIT=False)))
    monkeypatch.setattr(catalog, "import_module" if stage == "dependency" else "resolve_kernel", fail)
    with caplog.at_level("INFO"):
        assert catalog.resolve_engine("auto", algorithm="nsgaiii") == "numpy"
    assert "Numba could not be loaded" in caplog.text


def test_disabled_jit_selects_numpy_without_loading_kernels(monkeypatch):
    monkeypatch.setattr(catalog, "import_module", lambda _: SimpleNamespace(config=SimpleNamespace(DISABLE_JIT=True)))
    monkeypatch.setattr(catalog, "resolve_kernel", lambda _: pytest.fail("JIT is disabled"))
    assert catalog.resolve_engine("auto", algorithm="smpso") == "numpy"


@pytest.mark.parametrize("available", [False, True])
def test_auto_does_not_retry_objective_errors(monkeypatch, available):
    if available:
        pytest.importorskip("numba")
    from vamos.foundation.problem.zdt1 import ZDT1Problem

    calls = 0

    def fail(self, X, out):
        nonlocal calls
        calls += 1
        raise RuntimeError("objective failure")

    monkeypatch.setattr(catalog, "_has_numba", lambda: available)
    monkeypatch.setattr(ZDT1Problem, "evaluate", fail)
    with pytest.raises(RuntimeError, match="objective failure"):
        vamos.optimize("zdt1", engine="auto", algorithm="nsgaii", pop_size=8, max_evaluations=16)
    assert calls == 1


def test_auto_does_not_retry_compilation_errors(monkeypatch):
    pytest.importorskip("numba")
    from vamos.foundation.kernel.numba_backend import NumbaKernel

    def fail(*args, **kwargs):
        raise RuntimeError("compilation failed after selection")

    monkeypatch.setattr(NumbaKernel, "nsga2_ranking", fail)
    with pytest.raises(RuntimeError, match="compilation failed after selection"):
        vamos.optimize("zdt1", engine="auto", algorithm="nsgaii", pop_size=8, max_evaluations=16)


@pytest.mark.parametrize("error", ["ImportError", "OSError", "RuntimeError", "AttributeError"])
def test_broken_optional_install_falls_back_end_to_end(error):
    """Fresh processes also exercise the NumPy single-pair/MOEA-D shortcuts."""
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent(
        f"""
        import builtins
        original_import = builtins.__import__
        def blocked(name, *args, **kwargs):
            if name == 'numba' or name.startswith('numba.'):
                raise {error}('simulated incompatible optional dependency')
            return original_import(name, *args, **kwargs)
        builtins.__import__ = blocked
        import vamos
        import numpy as np
        from vamos.experiment.runtime import catalog
        # import_module bypasses __import__ for the initial name.
        original_import_module = catalog.import_module
        def blocked_module(name):
            if name == 'numba':
                raise {error}('simulated incompatible optional dependency')
            return original_import_module(name)
        catalog.import_module = blocked_module
        from vamos.engine.algorithm.registry import get_builtin_algorithm_names
        for algorithm in get_builtin_algorithm_names():
            kwargs = dict(algorithm=algorithm, seed=19, max_evaluations=300)
            reference = vamos.optimize('zdt1', engine='numpy', **kwargs)
            fallback = vamos.optimize('zdt1', engine='auto', **kwargs)
            assert fallback.meta['engine'] == 'numpy'
            assert fallback.meta['default_sources']['engine'] == 'auto'
            np.testing.assert_array_equal(reference.X, fallback.X)
            np.testing.assert_array_equal(reference.F, fallback.F)
        try:
            vamos.optimize('zdt1', algorithm='nsgaii', engine='numba', max_evaluations=300)
        except {error}:
            pass
        else:
            raise AssertionError('explicit numba must fail, never substitute')
        """
    )
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr


def test_saved_auto_study_freezes_backend_and_requested_config(tmp_path, monkeypatch):
    pytest.importorskip("numba")
    spec = vamos.StudySpec(problems=["zdt1"], algorithms=["spea2"], seeds=[7], engine="auto", pop_size=8, max_evaluations=24)
    plan = vamos.plan_study(spec)
    study = vamos.create_study(spec, output=tmp_path / "study")
    assert study.plan == plan.plan
    task = study.plan.tasks[0]
    assert task.requested_run["defaults"]["engine"] == "auto"
    assert task.resolved_run_spec["backend"]["kernel"]["resolution"]["name"] == "numba"
    monkeypatch.setattr(catalog, "_has_numba", lambda: False)
    loaded = vamos.load_study(tmp_path / "study")
    assert loaded.plan == study.plan
    completed = loaded.run()
    assert completed.manifest.counts.succeeded == 1


def test_auto_resolution_does_not_compile_or_draw_randomness(monkeypatch):
    pytest.importorskip("numba")
    import numpy as np

    from vamos.foundation.kernel.numba_backend import NumbaKernel

    def forbidden(*args, **kwargs):
        raise AssertionError("resolution must not execute kernels or consume RNG")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    monkeypatch.setattr(NumbaKernel, "nsga2_ranking", forbidden)
    monkeypatch.setattr(NumbaKernel, "variation_kernels", forbidden)
    assert catalog.resolve_engine("auto", algorithm="nsgaii") == "numba"


@pytest.mark.parametrize("problem", ["zdt1", "re21", "int_alloc", "bin_feat", "tsp6", "mixed_design"])
def test_auto_preserves_supported_encodings(problem):
    import numpy as np

    pytest.importorskip("numba")
    kwargs = dict(algorithm="nsgaii", pop_size=8, max_evaluations=40, seed=11)
    reference = vamos.optimize(problem, engine="numpy", **kwargs)
    auto = vamos.optimize(problem, engine="auto", **kwargs)
    np.testing.assert_array_equal(reference.X, auto.X)
    np.testing.assert_array_equal(reference.F, auto.F)
