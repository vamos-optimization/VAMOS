"""Exact parity of the compiled variation kernels with the NumPy reference operators."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba")

import vamos.engine.operators.impl.real.crossover_sbx as crossover_sbx_module
import vamos.foundation.kernel.operator_primitives as primitives
from vamos.engine.operators.impl.real import (
    BLXAlphaCrossover,
    PolynomialMutation,
    SBXCrossover,
    VariationWorkspace,
)
from vamos.engine.operators.impl.real.utils import with_variation_kernels
from vamos.foundation.kernel import _numba_variation
from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.foundation.kernel.numpy_backend import NumPyKernel

KERNELS = _numba_variation.NUMBA_VARIATION_KERNELS

requires_exact_power = pytest.mark.skipif(
    not _numba_variation.compiled_power_matches_numpy(),
    reason="compiled pow differs from np.power on this platform; compiled variation stays disabled",
)


@pytest.fixture
def numpy_only_single_pair(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable the availability-gated compiled single-pair SBX so the reference is pure NumPy."""
    monkeypatch.setattr(crossover_sbx_module, "available_variation_kernels", lambda: None)
    monkeypatch.setattr(primitives, "available_variation_kernels", lambda: None)


def _mating_tensor(rng: np.random.Generator, n_pairs: int, n_var: int, duplicate_fraction: float):
    lower = rng.uniform(-2.0, 0.0, n_var)
    upper = lower + rng.uniform(0.1, 3.0, n_var)
    parents = rng.uniform(lower, upper, size=(n_pairs, 2, n_var))
    duplicates = rng.random(n_pairs) < duplicate_fraction
    parents[duplicates, 1, :] = parents[duplicates, 0, :]
    at_bound = rng.random((n_pairs, 2)) < 0.25
    parents[:, :, 0] = np.where(at_bound, lower[0], parents[:, :, 0])
    parents[:, :, -1] = np.where(at_bound[:, ::-1], upper[-1], parents[:, :, -1])
    return parents, lower, upper


@requires_exact_power
@pytest.mark.numba
@pytest.mark.parametrize("use_workspace", [True, False])
@pytest.mark.parametrize("prob_var", [0.5, 1.0])
@pytest.mark.parametrize("eta", [20.0, 1.0, 0.0, 5.5])
@pytest.mark.parametrize("n_pairs", [1, 2, 9, 64])
def test_compiled_sbx_matches_numpy_values_and_generator_state(numpy_only_single_pair, use_workspace, prob_var, eta, n_pairs):
    for seed in range(25):
        parents, lower, upper = _mating_tensor(np.random.default_rng(seed), n_pairs, 7, duplicate_fraction=0.3)
        reference = SBXCrossover(
            prob_crossover=0.9,
            eta=eta,
            prob_var=prob_var,
            lower=lower,
            upper=upper,
            workspace=VariationWorkspace() if use_workspace else None,
        )
        compiled = SBXCrossover(
            prob_crossover=0.9,
            eta=eta,
            prob_var=prob_var,
            lower=lower,
            upper=upper,
            workspace=VariationWorkspace() if use_workspace else None,
            variation_kernels=KERNELS,
        )
        rng_ref = np.random.default_rng(1000 + seed)
        rng_cmp = np.random.default_rng(1000 + seed)
        expected = reference(parents.copy(), rng_ref)
        actual = compiled(parents.copy(), rng_cmp)
        np.testing.assert_array_equal(actual, expected)
        assert rng_cmp.bit_generator.state == rng_ref.bit_generator.state


@requires_exact_power
@pytest.mark.numba
def test_compiled_sbx_skips_draws_like_numpy_when_no_variable_changes(numpy_only_single_pair):
    parents = np.tile(np.linspace(0.1, 0.9, 5), (3, 2, 1))  # every pair holds identical parents
    lower, upper = np.zeros(5), np.ones(5)
    reference = SBXCrossover(prob_crossover=1.0, lower=lower, upper=upper, workspace=VariationWorkspace())
    compiled = SBXCrossover(prob_crossover=1.0, lower=lower, upper=upper, workspace=VariationWorkspace(), variation_kernels=KERNELS)
    rng_ref, rng_cmp = np.random.default_rng(3), np.random.default_rng(3)
    np.testing.assert_array_equal(compiled(parents.copy(), rng_cmp), reference(parents.copy(), rng_ref))
    assert rng_cmp.bit_generator.state == rng_ref.bit_generator.state
    # pair draws (3) + variable draws (15); no spread/swap grids because nothing changes
    expected = np.random.default_rng(3)
    expected.random(3 + 15)
    assert rng_cmp.bit_generator.state == expected.bit_generator.state


@requires_exact_power
@pytest.mark.numba
@pytest.mark.parametrize("use_workspace", [True, False])
def test_single_pair_sbx_with_numba_installed_matches_pure_numpy_stream(monkeypatch, use_workspace):
    """The availability-gated single-pair path must not change results or draws."""
    lower, upper = np.zeros(6), np.ones(6)
    for seed in range(60):
        rng = np.random.default_rng(seed)
        pair = rng.random((1, 2, 6))
        if seed % 3 == 0:
            pair[0, 1] = pair[0, 0]
        operator = SBXCrossover(prob_crossover=0.9, lower=lower, upper=upper, workspace=VariationWorkspace() if use_workspace else None)
        rng_gated = np.random.default_rng(seed + 7)
        gated = operator(pair.copy(), rng_gated)
        follow_gated = rng_gated.random(4)

        with monkeypatch.context() as patch:
            patch.setattr(crossover_sbx_module, "available_variation_kernels", lambda: None)
            patch.setattr(primitives, "available_variation_kernels", lambda: None)
            rng_numpy = np.random.default_rng(seed + 7)
            pure = operator(pair.copy(), rng_numpy)
            follow_numpy = rng_numpy.random(4)

        np.testing.assert_array_equal(gated, pure)
        np.testing.assert_array_equal(follow_gated, follow_numpy)


@requires_exact_power
@pytest.mark.numba
@pytest.mark.parametrize("use_workspace", [True, False])
@pytest.mark.parametrize("eta", [20.0, 15.0, 1.0, 0.0, 100.0])
@pytest.mark.parametrize("n_rows", [1, 3, 40])
def test_compiled_polynomial_mutation_matches_numpy(use_workspace, eta, n_rows):
    for seed in range(25):
        rng = np.random.default_rng(seed)
        lower = rng.uniform(-2.0, 0.0, 11)
        upper = lower + rng.uniform(0.0, 3.0, 11)
        upper[3] = lower[3]  # zero-width variable is never mutated
        X = rng.uniform(lower, upper, size=(n_rows, 11))
        X[:, 0] = lower[0]
        reference = PolynomialMutation(0.4, eta, lower=lower, upper=upper, workspace=VariationWorkspace() if use_workspace else None)
        compiled = PolynomialMutation(
            0.4,
            eta,
            lower=lower,
            upper=upper,
            workspace=VariationWorkspace() if use_workspace else None,
            variation_kernels=KERNELS,
        )
        rng_ref, rng_cmp = np.random.default_rng(seed), np.random.default_rng(seed)
        np.testing.assert_array_equal(compiled(X, rng_cmp), reference(X, rng_ref))
        assert rng_cmp.bit_generator.state == rng_ref.bit_generator.state


@pytest.mark.numba
def test_power_flag_reproduces_numpy_scalar_exponent_two():
    probe = np.linspace(0.5, 3.0, 4096)
    if _numba_variation.NUMPY_POWER_SQUARES:
        np.testing.assert_array_equal(np.power(probe, 2.0), probe * probe)
    # eta == 1 uses exponent 2.0 inside polynomial mutation; it must stay exact.
    X = np.random.default_rng(4).random((64, 9))
    for kernel in (NumPyKernel(), NumbaKernel()):
        assert kernel.variation_kernels() in (None, KERNELS)
    a, b = X.copy(), X.copy()
    NumPyKernel().polynomial_mutation(a, {"prob": 0.6, "eta": 1.0}, np.random.default_rng(9), 0.0, 1.0)
    NumbaKernel().polynomial_mutation(b, {"prob": 0.6, "eta": 1.0}, np.random.default_rng(9), 0.0, 1.0)
    np.testing.assert_array_equal(b, a)


@pytest.mark.numba
@pytest.mark.parametrize("n_rows", [0, 1, 2, 7, 40])
def test_numba_kernel_flat_sbx_matches_numpy_kernel(n_rows):
    for seed in range(20):
        X = np.random.default_rng(seed).random((n_rows, 6))
        if n_rows >= 2:
            X[1] = X[0]
        params = {"prob": 0.8, "eta": 15.0}
        expected = NumPyKernel().sbx_crossover(X, params, np.random.default_rng(seed), 0.0, 1.0)
        actual = NumbaKernel().sbx_crossover(X, params, np.random.default_rng(seed), 0.0, 1.0)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.numba
def test_variation_kernels_bind_only_to_opted_in_operators():
    lower, upper = np.zeros(3), np.ones(3)
    assert with_variation_kernels(SBXCrossover, {}, KERNELS) == {"variation_kernels": KERNELS}
    assert with_variation_kernels(PolynomialMutation, {}, KERNELS) == {"variation_kernels": KERNELS}
    assert with_variation_kernels(BLXAlphaCrossover, {}, KERNELS) == {}
    assert with_variation_kernels(SBXCrossover, {}, None) == {}
    BLXAlphaCrossover(lower=lower, upper=upper, **with_variation_kernels(BLXAlphaCrossover, {}, KERNELS))


@pytest.mark.numba
def test_compiled_variation_kernels_are_nopython():
    lower, upper = np.zeros(4), np.ones(4)
    parents = np.random.default_rng(0).random((5, 2, 4))
    KERNELS.sbx_crossover_pairs(parents, rng=np.random.default_rng(1), lower=lower, upper=upper, prob_crossover=1.0, eta=20.0, prob_var=0.5)
    X = np.random.default_rng(2).random((5, 4))
    KERNELS.polynomial_mutation(
        X, lower=lower, upper=upper, prob_mutation=1.0, eta=20.0, rnd_mask=np.zeros((5, 4)), rnd_delta=np.full((5, 4), 0.3)
    )
    for dispatcher in (_numba_variation._sbx_mark_active, _numba_variation._sbx_apply, _numba_variation.polynomial_mutation_apply):
        assert dispatcher.signatures
        assert dispatcher.nopython_signatures


@pytest.mark.numba
def test_variation_kernels_fall_back_to_numpy_when_power_differs(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_numba_variation, "compiled_power_matches_numpy", lambda: False)
    assert _numba_variation.verified_variation_kernels() is None
    kernel = NumbaKernel()
    assert kernel.variation_kernels() is None

    X = np.random.default_rng(3).random((16, 5))
    expected, actual = X.copy(), X.copy()
    NumPyKernel().polynomial_mutation(expected, {"prob": 0.5, "eta": 1.0}, np.random.default_rng(8), 0.0, 1.0)
    kernel.polynomial_mutation(actual, {"prob": 0.5, "eta": 1.0}, np.random.default_rng(8), 0.0, 1.0)
    np.testing.assert_array_equal(actual, expected)

    parents = np.random.default_rng(4).random((8, 5))
    np.testing.assert_array_equal(
        kernel.sbx_crossover(parents, {"prob": 1.0, "eta": 15.0}, np.random.default_rng(5), 0.0, 1.0),
        NumPyKernel().sbx_crossover(parents, {"prob": 1.0, "eta": 15.0}, np.random.default_rng(5), 0.0, 1.0),
    )


@pytest.mark.numba
def test_power_probe_matches_this_platform_numpy():
    # The probe must agree with a direct comparison on the exponents SBX/PM use by default.
    bases = np.linspace(0.0, 2.0, 513)
    out = np.empty_like(bases)
    matches = True
    with np.errstate(all="ignore"):
        for exponent in (21.0, -21.0, 1.0 / 21.0, 2.0, 0.5):
            _numba_variation._numpy_power_vector(bases, exponent, _numba_variation.NUMPY_POWER_SQUARES, out)
            matches &= bool(np.array_equal(out, np.power(bases, exponent), equal_nan=True))
    if _numba_variation.compiled_power_matches_numpy():
        assert matches


def _reference_differential_children(parents: np.ndarray, f: float, cr: float, rng: np.random.Generator) -> np.ndarray:
    # Verbatim NumPy branch of the MOEA/D DE crossover (vamos.engine.operators.policies.moead).
    n_pairs, _, n_vars = parents.shape
    base, p1, p2 = parents[:, 2, :], parents[:, 0, :], parents[:, 1, :]
    mutant = base + f * (p1 - p2)
    rand = rng.random((n_pairs, n_vars))
    mask = rand < cr
    j_rand = rng.integers(0, n_vars, size=n_pairs)
    mask[np.arange(n_pairs), j_rand] = True
    return np.where(mask, mutant, base)


@requires_exact_power
@pytest.mark.numba
@pytest.mark.parametrize("cr", [0.0, 0.3, 1.0])
def test_compiled_differential_crossover_matches_numpy(cr: float) -> None:
    for seed in range(50):
        parents = np.random.default_rng(seed).uniform(-3.0, 3.0, size=(int(seed % 7) + 1, 3, 9))
        rng_ref, rng_cmp = np.random.default_rng(seed + 100), np.random.default_rng(seed + 100)
        expected = _reference_differential_children(parents, 0.5, cr, rng_ref)
        rand = rng_cmp.random((parents.shape[0], parents.shape[2]))
        j_rand = rng_cmp.integers(0, parents.shape[2], size=parents.shape[0])
        actual = KERNELS.differential_crossover(parents, 0.5, cr, rand, j_rand)
        np.testing.assert_array_equal(actual, expected)
        assert rng_cmp.bit_generator.state == rng_ref.bit_generator.state
