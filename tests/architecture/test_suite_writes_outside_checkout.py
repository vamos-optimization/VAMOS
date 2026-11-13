"""The suite's caches stay outside the checkout (VR-134)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_numba_caches_compiled_kernels_outside_the_checkout() -> None:
    numba = pytest.importorskip("numba")
    # Without NUMBA_CACHE_DIR, Numba writes its cache next to the source, under src/**/__pycache__.
    cache_dir = numba.config.CACHE_DIR
    assert cache_dir
    assert not Path(cache_dir).resolve().is_relative_to(REPO_ROOT)


@pytest.mark.parametrize("gate", ["test_ruff_gate.py", "test_ruff_format_gate.py"])
def test_ruff_gates_do_not_write_a_cache(gate: str) -> None:
    source = (REPO_ROOT / "tests" / "architecture" / gate).read_text(encoding="utf-8")
    assert '"--no-cache"' in source
