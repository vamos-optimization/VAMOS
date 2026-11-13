import subprocess
import sys

import pytest


def _run_cmd(cmd, *, timeout: int = 30):
    return subprocess.run(cmd, shell=True, capture_output=True, timeout=timeout)


def test_self_check_module_smoke():
    # A full Windows extras environment can spend over 30 seconds collecting
    # package/environment metadata for the five tiny self-check runs.
    proc = _run_cmd(f"{sys.executable} -m vamos.experiment.diagnostics.self_check", timeout=120)
    assert proc.returncode == 0, proc.stderr.decode()


def test_vamos_benchmark_help():
    proc = _run_cmd(f"{sys.executable} -m vamos.experiment.benchmark.cli --help")
    assert proc.returncode == 0


@pytest.mark.cli
def test_vamos_zoo_help():
    proc = _run_cmd(f"{sys.executable} -m vamos.experiment.zoo.cli --help")
    assert proc.returncode == 0


@pytest.mark.studio
def test_vamos_studio_help():
    import importlib.util

    if importlib.util.find_spec("streamlit") is None:
        pytest.skip("streamlit not installed")
    proc = _run_cmd(f"{sys.executable} -m vamos.ux.studio.app --help")
    assert proc.returncode == 0


def test_cli_version_flag():
    from vamos.foundation.version import get_version

    proc = subprocess.run([sys.executable, "-m", "vamos.experiment.cli.main", "--version"], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"vamos {get_version()}"
