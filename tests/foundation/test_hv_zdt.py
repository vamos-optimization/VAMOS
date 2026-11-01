import numpy as np

from vamos.foundation.quality_indicators.hv_zdt import compute_normalized_hv


def test_normalized_hv_handles_points_above_default_reference():
    # Points exceed the default [1.1, 1.1] reference; function should auto-expand ref.
    F = np.array([[1.2, 1.0], [1.05, 1.3]])
    hv = compute_normalized_hv(F, "zdt1")
    assert hv >= 0.0
    assert hv <= 1.0 + 1e-6


def test_normalized_hv_zdt5_reference_front_is_one():
    from vamos.foundation.quality_indicators.hv_zdt import get_zdt_reference_front

    pf = get_zdt_reference_front("zdt5")
    hv = compute_normalized_hv(pf, "zdt5")
    assert np.isclose(hv, 1.0)


def _packaged_front(name):
    from vamos.foundation.problem.resolver import resolve_reference_front_path

    path = resolve_reference_front_path(name, None)
    return np.loadtxt(path, delimiter=",", ndmin=2)


def _igd(front, reference):
    distances = np.linalg.norm(reference[:, None, :] - front[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def test_zdt6_reference_front_matches_analytic_front():
    from vamos.foundation.quality_indicators.hv_zdt import get_zdt_reference_front

    front = get_zdt_reference_front("zdt6")

    np.testing.assert_allclose(front[:, 1], 1.0 - np.square(front[:, 0]))
    assert np.isclose(front[:, 0].min(), 0.2807753188, atol=1e-9)
    assert np.isclose(front[:, 0].max(), 1.0)
    assert _igd(front, _packaged_front("zdt6")) < 1e-3


def test_zdt3_reference_front_is_non_dominated():
    from vamos.foundation.quality_indicators.hv_zdt import get_zdt_reference_front
    from vamos.foundation.quality_indicators.pareto import pareto_filter

    front = get_zdt_reference_front("zdt3")

    assert pareto_filter(front).shape == front.shape
    assert _igd(front, _packaged_front("zdt3")) < 1e-3
