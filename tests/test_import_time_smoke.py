from __future__ import annotations

import json
import subprocess
import sys

BLACKLIST = (
    "torch",
    "tensorflow",
    "sklearn",
    "pymoo",
    "jmetalpy",
    "pygmo",
    "streamlit",
)


def _collect_imported_modules() -> list[str]:
    cmd = [
        sys.executable,
        "-c",
        "\n".join(
            [
                "import json, sys",
                "mods = ['vamos', 'vamos.api', 'vamos.algorithms', 'vamos.ux.api']",
                "for m in mods:",
                "    __import__(m)",
                "print(json.dumps(sorted(sys.modules.keys())))",
            ]
        ),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise AssertionError(f"Import-time smoke failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    stdout_lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        raise AssertionError("Import-time smoke produced no output.")
    payload = stdout_lines[-1]
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            f"Failed to parse sys.modules payload:\n{payload}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        ) from exc


def _find_blacklisted_modules(modules: list[str]) -> list[str]:
    offenders = []
    for banned in BLACKLIST:
        for name in modules:
            if name == banned or name.startswith(f"{banned}."):
                offenders.append(name)
                break
    return sorted(set(offenders))


def test_import_time_smoke() -> None:
    modules = _collect_imported_modules()
    offenders = _find_blacklisted_modules(modules)
    if offenders:
        raise AssertionError(f"Import-time smoke loaded optional heavy modules: {offenders}")


def test_foundation_import_does_not_load_experiment() -> None:
    """Importing a foundation module loads neither the experiment layer nor the algorithms (VR-100).

    ``import vamos.foundation.encoding`` runs ``vamos/__init__.py``, whose facade names now load on
    first access.
    """
    cmd = [
        sys.executable,
        "-c",
        "import json, sys; import vamos.foundation.encoding; print(json.dumps(sorted(sys.modules)))",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    modules = json.loads(result.stdout.splitlines()[-1])
    loaded = [name for name in modules if name.startswith(("vamos.experiment", "vamos.engine.algorithm"))]
    assert loaded == []


def test_lazy_facades_resolve_every_public_name() -> None:
    """Every name of ``__all__`` resolves, ``dir()`` lists it, and ``from vamos import *`` works."""
    import vamos
    import vamos.api

    for module in (vamos, vamos.api):
        for name in module.__all__:
            assert getattr(module, name) is not None, (module.__name__, name)
        assert set(module.__all__) <= set(dir(module))
    namespace: dict[str, object] = {}
    exec("from vamos import *", namespace)
    assert namespace["optimize"] is vamos.api.optimize


def test_subpackages_stay_reachable_after_import_vamos() -> None:
    """``vamos.api``, ``vamos.run_artifacts`` and the layers load as attributes, as with the eager facade."""
    cmd = [
        sys.executable,
        "-c",
        "import vamos; print(vamos.api.optimize.__name__, vamos.run_artifacts.load_run.__name__, vamos.experiment.__name__);"
        "print(hasattr(vamos, 'no_such_module'))",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert result.stdout.split() == ["optimize", "load_run", "vamos.experiment", "False"]
