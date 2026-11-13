"""Deterministic regression tests for repair helpers."""

import numpy as np
from numpy.testing import assert_allclose

from vamos.engine.operators.impl.real import (
    ClampRepair,
    MidpointBaseRepair,
    ReflectRepair,
    ResampleRepair,
    RoundRepair,
    WrappingRepair,
)
from vamos.engine.operators.impl.repair import GradientRepair

VIOLATING = np.array(
    [
        [-2.5, 0.5, 2.5],
        [1.2, -0.4, -3.1],
        [3.0, 4.2, 1.2],
    ],
    dtype=float,
)

LOWER = np.array([-1.5, 0.0, -2.0])
UPPER = np.array([1.5, 3.5, 2.0])


def test_clamp_repair_limits_values_to_bounds():
    operator = ClampRepair()
    rng = np.random.default_rng(0)
    repaired = operator(VIOLATING, LOWER, UPPER, rng)
    expected = np.array(
        [
            [-1.5, 0.5, 2.0],
            [1.2, 0.0, -2.0],
            [1.5, 3.5, 1.2],
        ]
    )
    assert_allclose(repaired, expected)


def test_reflect_repair_reflects_out_of_bounds_entries():
    operator = ReflectRepair()
    rng = np.random.default_rng(0)
    repaired = operator(VIOLATING, LOWER, UPPER, rng)
    expected = np.array(
        [
            [-0.5, 0.5, 1.5],
            [1.2, 0.4, -0.9],
            [0.0, 2.8, 1.2],
        ]
    )
    assert_allclose(repaired, expected)


def test_resample_repair_resamples_violated_genes():
    operator = ResampleRepair()
    rng = np.random.default_rng(11)
    repaired = operator(VIOLATING, LOWER, UPPER, rng)
    expected = np.array(
        [
            [-1.1142893917, 0.5, 0.4059934305],
            [1.2, 0.517741296, 1.7128440918],
            [-1.2887382715, 0.4542088229, 1.2],
        ]
    )
    assert_allclose(repaired, expected)


def test_round_repair_rounds_and_clamps():
    operator = RoundRepair()
    rng = np.random.default_rng(0)
    repaired = operator(VIOLATING, LOWER, UPPER, rng)
    expected = np.array(
        [
            [-1.5, 0.0, 2.0],
            [1.0, 0.0, -2.0],
            [1.5, 3.5, 1.0],
        ]
    )
    assert_allclose(repaired, expected)


def test_wrapping_repair_wraps_toroidally():
    operator = WrappingRepair()
    rng = np.random.default_rng(0)
    repaired = operator(VIOLATING, LOWER, UPPER, rng)
    # width = [3.0, 3.5, 4.0]
    # [-2.5, 0.5, 2.5]: gene0: (-2.5 - (-1.5)) % 3.0 = (-1.0) % 3.0 = 2.0 → -1.5+2.0=0.5
    #                    gene1: (0.5 - 0.0) % 3.5 = 0.5 → 0.0+0.5=0.5
    #                    gene2: (2.5 - (-2.0)) % 4.0 = 4.5 % 4.0 = 0.5 → -2.0+0.5=-1.5
    # [1.2, -0.4, -3.1]: gene0: (1.2-(-1.5)) % 3.0 = 2.7 % 3.0 = 2.7 → -1.5+2.7=1.2
    #                     gene1: (-0.4-0.0) % 3.5 = -0.4 % 3.5 = 3.1 → 0.0+3.1=3.1
    #                     gene2: (-3.1-(-2.0)) % 4.0 = -1.1 % 4.0 = 2.9 → -2.0+2.9=0.9
    # [3.0, 4.2, 1.2]: gene0: (3.0-(-1.5)) % 3.0 = 4.5 % 3.0 = 1.5 → -1.5+1.5=0.0
    #                   gene1: (4.2-0.0) % 3.5 = 0.7 → 0.0+0.7=0.7
    #                   gene2: (1.2-(-2.0)) % 4.0 = 3.2 % 4.0 = 3.2 → -2.0+3.2=1.2
    expected = np.array(
        [
            [0.5, 0.5, -1.5],
            [1.2, 3.1, 0.9],
            [0.0, 0.7, 1.2],
        ]
    )
    assert_allclose(repaired, expected)


def test_wrapping_repair_in_bounds_unchanged():
    operator = WrappingRepair()
    rng = np.random.default_rng(0)
    in_bounds = np.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [1.5, 3.5, 2.0],
            [-1.5, 0.0, -2.0],
        ]
    )
    repaired = operator(in_bounds, LOWER, UPPER, rng)
    assert_allclose(repaired, in_bounds)


def test_midpoint_base_repair_moves_halfway():
    operator = MidpointBaseRepair()
    rng = np.random.default_rng(0)
    repaired = operator(VIOLATING, LOWER, UPPER, rng)
    # midpoint = [0.0, 1.75, 0.0]
    # low target = (lower + midpoint) / 2 = [-0.75, 0.875, -1.0]
    # high target = (midpoint + upper) / 2 = [0.75, 2.625, 1.0]
    expected = np.array(
        [
            [-0.75, 0.5, 1.0],
            [1.2, 0.875, -1.0],
            [0.75, 2.625, 1.2],
        ]
    )
    assert_allclose(repaired, expected)


def test_midpoint_base_repair_less_extreme_than_clamp():
    """Midpoint-base repair should move violations to an interior feasible point."""
    operator = MidpointBaseRepair()
    rng = np.random.default_rng(0)
    x = np.array([[2.0, -1.0, 3.0]])
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([1.0, 1.0, 1.0])
    repaired = operator(x, lower, upper, rng)
    expected = np.array([[0.75, 0.25, 0.75]])
    assert_allclose(repaired, expected)
    assert np.all(repaired >= lower)
    assert np.all(repaired <= upper)


def test_gradient_repair_reduces_violations():
    """GradientRepair should reduce constraint violations via gradient steps."""

    def constraint_fun(X: np.ndarray) -> np.ndarray:
        # Single constraint: x0 + x1 - 1.0 <= 0
        return np.maximum(0.0, X[:, 0:1] + X[:, 1:2] - 1.0)

    def constraint_grad(X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        grads = np.zeros((n, 1, X.shape[1]))
        grads[:, 0, 0] = 1.0
        grads[:, 0, 1] = 1.0
        return grads

    repair = GradientRepair(constraint_fun, constraint_grad, step_size=0.5, max_iters=10)
    X = np.array([[0.8, 0.8], [0.3, 0.2]])  # first violates, second doesn't
    xl = np.array([0.0, 0.0])
    xu = np.array([1.0, 1.0])
    rng = np.random.default_rng(0)

    repaired = repair(X, xl, xu, rng)

    # Second point should be unchanged (no violation)
    assert_allclose(repaired[1], [0.3, 0.2])
    # First point's violation should be reduced
    violation_before = max(0.0, X[0, 0] + X[0, 1] - 1.0)
    violation_after = max(0.0, repaired[0, 0] + repaired[0, 1] - 1.0)
    assert violation_after < violation_before


def test_gradient_repair_respects_bounds():
    """GradientRepair must keep solutions within [xl, xu]."""

    def constraint_fun(X: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, X[:, 0:1] - 0.2)

    def constraint_grad(X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        grads = np.zeros((n, 1, X.shape[1]))
        grads[:, 0, 0] = 1.0
        return grads

    repair = GradientRepair(constraint_fun, constraint_grad, step_size=10.0, max_iters=5)
    X = np.array([[0.5, 0.5]])
    xl = np.array([0.0, 0.0])
    xu = np.array([1.0, 1.0])

    repaired = repair(X, xl, xu)
    assert np.all(repaired >= xl)
    assert np.all(repaired <= xu)


def test_registry_resolves_new_repairs():
    """All new repair names should be resolvable from the operator registry."""
    from vamos.engine.operators.impl.registry import get_operator_registry

    reg = get_operator_registry()
    assert reg.get("wrap") is WrappingRepair
    assert reg.get("wrapping") is WrappingRepair
    assert reg.get("midpoint") is MidpointBaseRepair
    assert reg.get("midpoint_base") is MidpointBaseRepair
    assert reg.get("gradient") is GradientRepair


def test_smpso_resolve_repair_supports_new_repairs():
    from vamos.engine.algorithm.smpso.helpers import resolve_repair

    def constraint_fun(X: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, X[:, 0:1] - 0.5)

    def constraint_grad(X: np.ndarray) -> np.ndarray:
        grads = np.zeros((X.shape[0], 1, X.shape[1]))
        grads[:, 0, 0] = 1.0
        return grads

    assert isinstance(resolve_repair(("wrap", {})), WrappingRepair)
    assert isinstance(resolve_repair(("midpoint", {})), MidpointBaseRepair)
    assert isinstance(
        resolve_repair(
            (
                "gradient",
                {"constraint_fun": constraint_fun, "constraint_grad": constraint_grad},
            )
        ),
        GradientRepair,
    )


def test_gradient_repair_ignores_satisfied_constraints():
    """A satisfied constraint (g < 0) must not offset a violated one."""

    def constraint_fun(X: np.ndarray) -> np.ndarray:
        # g1 = x0 - 0.2 is violated by 0.3 at x0 = 0.5; g2 = -0.8 is satisfied.
        return np.column_stack([X[:, 0] - 0.2, np.full(X.shape[0], -0.8)])

    def constraint_grad(X: np.ndarray) -> np.ndarray:
        grads = np.zeros((X.shape[0], 2, X.shape[1]))
        grads[:, 0, 0] = 1.0
        return grads

    repair = GradientRepair(constraint_fun, constraint_grad, step_size=0.5, max_iters=20)
    repaired = repair(np.array([[0.5]]), np.array([0.0]), np.array([1.0]))

    assert repaired[0, 0] < 0.2 + 1e-3


def test_reflect_and_wrapping_repairs_send_infinities_to_the_bounds():
    X = np.array([[np.inf, -np.inf, 0.5]])
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([1.0, 2.0, 1.0])
    rng = np.random.default_rng(0)

    for repair in (ReflectRepair(), WrappingRepair()):
        repaired = repair(X.copy(), lower, upper, rng)
        assert_allclose(repaired, [[1.0, 0.0, 0.5]], err_msg=type(repair).__name__)
