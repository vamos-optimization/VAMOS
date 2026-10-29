"""Compiled steady-state SMS-EMOA kernels against their NumPy references."""

from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("numba")

from vamos.engine.algorithm.smsemoa.helpers import refresh_selection_cache, survival_selection
from vamos.engine.algorithm.smsemoa.state import SMSEMOAState
from vamos.engine.operators.impl.real import ClampRepair, PolynomialMutation, SBXCrossover, VariationWorkspace
from vamos.engine.operators.policies.real_repair import apply_policy_repair
from vamos.foundation.constraints.utils import compute_violation
from vamos.foundation.kernel._numba_single_child import npy_clip
from vamos.foundation.kernel._numba_smsemoa_measures import (
    front_contributions_2d,
    front_contributions_2d_match_reference,
    row_violations,
)
from vamos.foundation.kernel.backend import SMSEMOA_CONTRIBUTIONS, SMSEMOA_DELEGATE
from vamos.foundation.kernel.numba_backend import NumbaKernel
from vamos.foundation.quality_indicators import hypervolume as hv

pytestmark = pytest.mark.numba


def _front_2d(rng: np.random.Generator, size: int) -> np.ndarray:
    points = np.column_stack((np.sort(rng.random(size)), np.sort(rng.random(size))[::-1]))
    if size > 2 and rng.random() < 0.5:
        points[int(rng.integers(0, size))] = points[int(rng.integers(0, size))]
    return np.ascontiguousarray(points[rng.permutation(size)] * rng.choice((1.0e-3, 1.0, 5.0e2)))


def test_clip_follows_numpy_signed_zero_and_nan_rules() -> None:
    values = np.array([-0.0, 0.0, -1.0, 2.0, 0.5, 1.0, np.inf, -np.inf, np.nan, 1.0e-300, -1.0e-300])
    for lower, upper in ((0.0, 1.0), (-0.0, 1.0), (0.0, -0.0), (-0.0, 0.0), (0.0, 0.0), (-1.0, -0.0)):
        expected = np.clip(values, np.full(values.size, lower), np.full(values.size, upper))
        actual = np.array([npy_clip(v, lower, upper) for v in values])
        assert actual.tobytes() == expected.tobytes()


def test_front_contributions_match_every_reference_bit_for_bit() -> None:
    assert front_contributions_2d_match_reference()
    rng = np.random.default_rng(0)
    moocore = pytest.importorskip("moocore") if hv._MOOCORE is not None else None
    for _ in range(400):
        points = _front_2d(rng, int(rng.integers(2, 60)))
        ref = points.max(axis=0) + float(rng.choice((0.0, 0.1, 1.0)))
        ok, actual = front_contributions_2d(points, ref)
        assert ok
        assert actual.tobytes() == hv._hypervolume_contributions_2d(points, ref).tobytes()
        assert actual.tobytes() == np.asarray(hv.hypervolume_contributions(points, ref)).tobytes()
        if moocore is not None:
            assert actual.tobytes() == np.asarray(moocore.hv_contributions(points, ref=ref)).tobytes()


@pytest.mark.parametrize("n_constraints", [1, 2, 7, 8, 9, 16, 17, 40, 128])
def test_violation_sums_follow_numpy_summation_order(n_constraints: int) -> None:
    rng = np.random.default_rng(n_constraints)
    G = rng.standard_normal((300, n_constraints)) * rng.choice((1.0e-3, 1.0, 1.0e3), size=(300, n_constraints))
    np.testing.assert_array_equal(row_violations(np.ascontiguousarray(G)), compute_violation(G))


@pytest.mark.parametrize("prob,prob_var", [(1.0, 0.5), (0.7, 0.5), (1.0, 1.0), (0.0, 0.5)])
def test_single_child_reproduces_the_operator_chain(prob: float, prob_var: float) -> None:
    kernels = NumbaKernel().variation_kernels()
    if kernels is None:
        pytest.skip("compiled pow differs from np.power on this platform")
    n_var = 12
    lower, upper = np.zeros(n_var), np.linspace(1.0, 3.0, n_var)
    workspace = VariationWorkspace()
    crossover = SBXCrossover(
        prob, 15.0, prob_var, lower=lower, upper=upper, workspace=workspace, allow_inplace=True, variation_kernels=kernels
    )
    mutation = PolynomialMutation(0.3, 20.0, lower=lower, upper=upper, workspace=workspace, variation_kernels=kernels)
    child = kernels.single_child_sbx_polynomial(
        sbx_prob=prob, sbx_eta=15.0, sbx_prob_var=prob_var, sbx_lower=lower, sbx_upper=upper,
        pm_prob=0.3, pm_eta=20.0, pm_lower=lower, pm_upper=upper, clip_lower=lower, clip_upper=upper,
    )  # fmt: skip
    population = np.random.default_rng(1).random((10, n_var)) * upper
    population[3] = population[4]  # identical parents leave SBX without active variables
    reference_rng, compiled_rng = np.random.default_rng(9), np.random.default_rng(9)
    for first, second in [(0, 1), (3, 4), (5, 5), (9, 2)] * 5:
        offspring = apply_policy_repair(
            ClampRepair(), crossover(population[[first, second]].reshape(1, 2, n_var), reference_rng), lower, upper, reference_rng
        )
        expected = apply_policy_repair(
            ClampRepair(), mutation(offspring.reshape(-1, n_var)[0:1], reference_rng), lower, upper, reference_rng
        )
        out = np.empty((1, n_var))
        child(population, first, second, out, compiled_rng)
        assert out.tobytes() == expected.tobytes()
        assert compiled_rng.bit_generator.state == reference_rng.bit_generator.state


def _random_state(rng: np.random.Generator, constrained: bool) -> tuple[SMSEMOAState, int, bool]:
    n_obj, n = int(rng.choice([2, 3])), int(rng.choice([2, 5, 20, 50]))
    grid = rng.random() < 0.4
    F = rng.integers(0, 4, size=(n, n_obj)).astype(float) if grid else rng.random((n, n_obj))
    G = rng.random((n, int(rng.choice([1, 3, 9])))) * 2.0 - 1.2 if constrained else None
    state = SMSEMOAState(
        X=rng.random((n, 3)),
        F=F,
        G=G,
        rng=np.random.default_rng(0),
        pop_size=n,
        constraint_mode="feasibility" if constrained else "none",
        ref_point=F.max(axis=0) + 1.0,
        ref_offset=float(rng.choice([1.0, 0.1])),
        ref_adaptive=True,
        eliminate_duplicates=bool(rng.random() < 0.3),
    )
    return state, n_obj, grid


@pytest.mark.parametrize("constrained", [False, True], ids=["unconstrained", "constrained"])
def test_survival_step_matches_survival_selection(constrained: bool) -> None:
    kernel = NumbaKernel()
    rng = np.random.default_rng(42 + constrained)
    for _ in range(40):
        reference, n_obj, grid = _random_state(rng, constrained)
        refresh_selection_cache(reference, kernel)
        fast = copy.deepcopy(reference)
        n = fast.X.shape[0]
        n_con = 0 if fast.G is None else fast.G.shape[1]
        survival = kernel.selection_kernels().smsemoa_survival(
            pop_size=n, n_obj=n_obj, n_constraints=n_con, ref_offset=fast.ref_offset, ref_adaptive=True,
            eliminate_duplicates=fast.eliminate_duplicates, check_feasibility=constrained,
        )  # fmt: skip
        survival.reset(fast.F)
        G = fast.G if fast.G is not None else np.empty((n, 0))
        for _ in range(30):
            source = int(rng.integers(0, n))
            duplicate = rng.random() < 0.2
            x = fast.X[source : source + 1].copy() if duplicate else rng.random((1, 3))
            f = (
                fast.F[source : source + 1].copy()
                if duplicate
                else (rng.integers(0, 4, size=(1, n_obj)).astype(float) if grid else rng.random((1, n_obj)))
            )
            g = rng.random((1, n_con)) * 2.0 - 1.2 if constrained else None
            survival_selection(reference, x, f, g, kernel)
            g_row = g[0] if g is not None else np.empty(0)
            status, size = survival.insert(fast.X, fast.F, G, x[0], f[0], g_row, fast.ranks, fast.crowding, fast.ref_point)
            assert status != SMSEMOA_DELEGATE
            if status == SMSEMOA_CONTRIBUTIONS:
                worst = survival.worst[:size]
                remove = int(worst[int(np.argmin(kernel.hypervolume_contributions(survival.F_work[worst], fast.ref_point)))])
                if remove != n:
                    survival.commit(fast.X, fast.F, G, x[0], f[0], g_row, fast.ranks, fast.crowding, remove)
            for name in ("X", "F", "G", "crowding", "ref_point"):
                left, right = getattr(reference, name), getattr(fast, name)
                assert (left is None and right is None) or left.tobytes() == right.tobytes(), name
            np.testing.assert_array_equal(fast.ranks, reference.ranks)
