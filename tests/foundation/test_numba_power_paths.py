"""The compiled power follows NumPy's scalar-exponent fast paths, so the compiled variation stays on.

``np.power(x, e)`` with a scalar ``e`` evaluates some exponents without the C library's ``pow``:
NumPy 2 computes ``e = 2`` as ``x * x``, and NumPy 2.4 also ``e = 0.5`` as ``sqrt(x)`` and
``e = -1`` as ``1 / x``. These can differ from ``pow`` by one ULP (and ``sqrt(-0.0)`` is ``-0.0``
where ``pow`` gives ``+0.0``), so compiled code that calls ``pow`` differs from the NumPy reference
there, and the guard ``compiled_power_matches_numpy`` turns the compiled variation off. The probes
of ``numpy_power_paths`` detect each fast path at run time, and ``_numpy_power`` takes the same one.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

pytest.importorskip("numba")

from vamos.foundation.kernel import _numba_variation as nv
from vamos.foundation.kernel.numba_backend import NumbaKernel

pytestmark = pytest.mark.numba

_rng = np.random.default_rng(2024)
BASES = np.concatenate(
    (
        [0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 5e-324, 2.2250738585072014e-308, 1e-300, 1e300],
        np.linspace(-3.0, 3.0, 4097),
        np.geomspace(1e-300, 1e300, 4097),
        -np.geomspace(1e-300, 1e300, 1025),
        _rng.uniform(0.0, 2.0, 50_000),
        _rng.uniform(0.999, 1.001, 5_000),
    )
)
# exponent -> (bit of numpy_power_paths, the operation NumPy's fast path performs)
FAST_PATHS: dict[float, tuple[int, Callable[[np.ndarray], np.ndarray]]] = {
    2.0: (1, lambda x: x * x),
    0.5: (2, np.sqrt),
    -1.0: (4, lambda x: 1.0 / x),
}
# Exponents that no NumPy release special-cases, among those SBX and polynomial mutation use.
GENERAL = (21.0, -21.0, 1.0 / 21.0, 16.0, 1.0 / 16.0, 6.5, -6.5, 1.0 / 6.5, 3.0, -2.0, 1.5, 0.25)


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    """Equal values and signs of zero; NaNs equal whatever their payload."""
    nan = np.isnan(a)
    if not np.array_equal(nan, np.isnan(b)):
        return False
    return bool(np.array_equal(a[~nan], b[~nan]) and np.array_equal(np.signbit(a[~nan]), np.signbit(b[~nan])))


def _compiled(exponent: float, paths: int) -> np.ndarray:
    out = np.empty_like(BASES)
    with np.errstate(all="ignore"):
        nv._numpy_power_vector(BASES, exponent, paths, out)
    return out


def _numpy(exponent: float) -> np.ndarray:
    with np.errstate(all="ignore"):
        return np.asarray(np.power(BASES, exponent))


def test_the_guard_passes_where_numpy_calls_the_platform_pow() -> None:
    # With an exponent NumPy does not special-case the flags play no part, so 0 works with any version of the kernels.
    if not all(_same(_compiled(exponent, 0), _numpy(exponent)) for exponent in GENERAL):
        pytest.skip("NumPy's power is not the platform pow here (a vendor SIMD library); the guard keeps compiled variation off")
    assert nv.compiled_power_matches_numpy()
    assert NumbaKernel().variation_kernels() is not None


def test_the_flag_bits() -> None:
    assert (nv.POWER_SQUARE, nv.POWER_SQRT, nv.POWER_RECIPROCAL) == (1, 2, 4)


def test_the_probes_detect_numpys_fast_paths() -> None:
    paths = nv.numpy_power_paths()
    with np.errstate(all="ignore"):
        for exponent, (bit, fast) in FAST_PATHS.items():
            assert bool(paths & bit) == _same(_numpy(exponent), np.asarray(fast(BASES))), exponent


@pytest.mark.parametrize("exponent", sorted(FAST_PATHS))
def test_the_compiled_power_takes_numpys_fast_path(exponent: float) -> None:
    bit, fast = FAST_PATHS[exponent]
    with np.errstate(all="ignore"):
        if not _same(_numpy(exponent), np.asarray(fast(BASES))):
            pytest.skip(f"this NumPy evaluates the exponent {exponent} with pow")
    assert _same(_compiled(exponent, nv.numpy_power_paths()), _numpy(exponent))
    # Without the flag the compiled code calls pow, which differs from NumPy's fast path somewhere (or the path is moot).
    assert nv.numpy_power_paths() & bit


@pytest.mark.parametrize("exponent", GENERAL)
def test_the_flags_leave_the_other_exponents_to_pow(exponent: float) -> None:
    assert _same(_compiled(exponent, nv.numpy_power_paths()), _compiled(exponent, 0))
