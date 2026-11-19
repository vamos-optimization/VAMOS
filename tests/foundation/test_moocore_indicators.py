import itertools

import numpy as np
import pytest

from vamos.foundation.quality_indicators import hypervolume as hv
from vamos.foundation.quality_indicators import moocore_indicators as mi

pytestmark = pytest.mark.backends


@pytest.mark.skipif(not mi.has_moocore(), reason="MooCore not installed")
def test_hv_wrapper_matches_moocore():
    import moocore

    F = np.array([[0.2, 0.3], [0.4, 0.1]])
    ref = np.array([1.0, 1.0])

    wrapper = mi.HVIndicator(reference_point=ref)
    res = wrapper.compute(F)

    expected = moocore.hypervolume(F, ref=ref)
    assert np.isclose(res.value, expected)


@pytest.mark.skipif(not mi.has_moocore(), reason="MooCore not installed")
def test_igd_and_epsilon_are_zero_on_reference():
    ref = np.array([[1.0, 1.0], [2.0, 2.0]])
    approx = ref.copy()

    igd = mi.IGDIndicator(reference_front=ref).compute(approx).value
    igd_plus = mi.IGDPlusIndicator(reference_front=ref).compute(approx).value
    eps_add = mi.EpsilonAdditiveIndicator(reference_front=ref).compute(approx).value
    eps_mult = mi.EpsilonMultiplicativeIndicator(reference_front=ref).compute(approx).value

    assert igd == pytest.approx(0.0)
    assert igd_plus == pytest.approx(0.0)
    assert eps_add == pytest.approx(0.0)
    assert eps_mult == pytest.approx(1.0)


@pytest.mark.skipif(not mi.has_moocore(), reason="MooCore not installed")
def test_indicators_match_moocore_reference_implementation():
    import moocore

    approx = np.array([[3.5, 5.5], [3.6, 4.1], [4.1, 3.2], [5.5, 1.5]])
    ref_front = np.array([[1, 6], [2, 5], [3, 4], [4, 3], [5, 2], [6, 1]], dtype=float)

    assert mi.IGDIndicator(reference_front=ref_front).compute(approx).value == pytest.approx(moocore.igd(approx, ref=ref_front))
    assert mi.IGDPlusIndicator(reference_front=ref_front).compute(approx).value == pytest.approx(moocore.igd_plus(approx, ref=ref_front))
    assert mi.EpsilonAdditiveIndicator(reference_front=ref_front).compute(approx).value == pytest.approx(
        moocore.epsilon_additive(approx, ref=ref_front)
    )
    assert mi.EpsilonMultiplicativeIndicator(reference_front=ref_front).compute(approx).value == pytest.approx(
        moocore.epsilon_mult(approx, ref=ref_front)
    )
    assert mi.AvgHausdorffIndicator(reference_front=ref_front).compute(approx).value == pytest.approx(
        moocore.avg_hausdorff_dist(approx, ref_front)
    )


@pytest.mark.skipif(not mi.has_moocore(), reason="MooCore not installed")
def test_hv_wrapper_supports_maximise():
    import moocore

    F = np.array([[1.0, 1.0], [2.0, 0.5], [0.5, 2.0]])
    ref = np.array([0.0, 0.0])

    wrapper = mi.HVIndicator(reference_point=ref)
    res = wrapper.compute(F, maximise=True)
    assert res.value == pytest.approx(moocore.hypervolume(F, ref=ref, maximise=True))


_REFERENCE_FRONT = np.array([[0.1, 1.0], [0.5, 0.5], [1.0, 0.1]])
_REFERENCE_POINT = np.array([1.1, 1.1])


@pytest.mark.skipif(not mi.has_moocore(), reason="MooCore not installed")
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_indicators_reject_non_finite_points(bad: float) -> None:
    front = np.array([[bad, 0.1], [0.5, 0.5]])
    indicators = [
        mi.HVIndicator(reference_point=_REFERENCE_POINT),
        mi.HVContributionsIndicator(reference_point=_REFERENCE_POINT),
        mi.HVApproxIndicator(reference_point=_REFERENCE_POINT),
        mi.IGDIndicator(reference_front=_REFERENCE_FRONT),
        mi.IGDPlusIndicator(reference_front=_REFERENCE_FRONT),
        mi.EpsilonAdditiveIndicator(reference_front=_REFERENCE_FRONT),
        mi.EpsilonMultiplicativeIndicator(reference_front=_REFERENCE_FRONT),
        mi.AvgHausdorffIndicator(reference_front=_REFERENCE_FRONT),
    ]
    for indicator in indicators:
        with pytest.raises(ValueError, match="finite"):
            indicator.compute(front)
    with pytest.raises(ValueError, match="finite"):
        mi.IGDIndicator(reference_front=front).compute(_REFERENCE_FRONT)
    with pytest.raises(ValueError, match="finite"):
        mi.HVIndicator(reference_point=np.array([bad, 1.0])).compute(_REFERENCE_FRONT)


@pytest.mark.skipif(not mi.has_moocore(), reason="MooCore not installed")
def test_hv_indicators_require_reference_point() -> None:
    front = np.array([[1.0, 2.0], [2.0, 1.0]])
    indicators = [
        mi.HVIndicator(),
        mi.HVContributionsIndicator(),
        mi.HVApproxIndicator(),
        mi.WHVRectIndicator(rectangles=np.array([[0.0, 0.0, 3.0, 3.0, 1.0]])),
        mi.get_indicator("hv"),
    ]
    for indicator in indicators:
        with pytest.raises(ValueError, match="reference_point"):
            indicator.compute(front)
    with pytest.raises(ValueError, match="reference_point"):
        mi.HVIndicator(reference_point=np.array([3.0, 3.0, 3.0])).compute(front)


def test_outside_points_are_filtered_in_every_hv_path() -> None:
    points = np.array([[0.4, 0.2], [1.2, 0.1]])
    ref = np.array([1.0, 1.0])

    assert hv.hypervolume(points, ref) == pytest.approx(0.48)
    assert hv.compute_hypervolume(points, ref) == pytest.approx(0.48)
    assert hv.hypervolume_contributions(points, ref) == pytest.approx([0.48, 0.0])
    if mi.has_moocore():
        assert mi.HVIndicator(reference_point=ref).compute(points).value == pytest.approx(0.48)
        assert mi.HVContributionsIndicator(reference_point=ref).compute(points).value == pytest.approx([0.48, 0.0])


def _dtlz2_front(n_obj: int, divisions: int) -> np.ndarray:
    """DTLZ2's front: a Das-Dennis simplex lattice projected onto the unit sphere."""
    rows = [c for c in itertools.product(range(divisions + 1), repeat=n_obj - 1) if sum(c) <= divisions]
    w = np.array([(*c, divisions - sum(c)) for c in rows], dtype=float)
    return w / np.linalg.norm(w, axis=1, keepdims=True)


@pytest.mark.skipif(not mi.has_moocore(), reason="MooCore not installed")
@pytest.mark.parametrize(("method", "samples"), [(None, 10_000), ("Rphi-FWE+", 10_000), ("DZ2019-HW", 10_000), ("DZ2019-MC", 100_000)])
def test_hv_approx_agrees_with_the_exact_hypervolume_on_dtlz2_with_four_objectives(method: str | None, samples: int) -> None:
    # Development after 1.4.0: 286 points of DTLZ2's front with four objectives.
    front = _dtlz2_front(4, 10)
    ref = np.full(4, 1.1)
    exact = mi.HVIndicator(reference_point=ref).compute(front).value

    approx = mi.HVApproxIndicator(reference_point=ref).compute(front, samples=samples, seed=7, method=method)

    assert approx.value == pytest.approx(exact, rel=0.01)
    assert approx.details["samples"] == samples


@pytest.mark.skipif(not mi.has_moocore(), reason="MooCore not installed")
def test_hv_approx_passes_its_seed_to_moocore() -> None:
    front = _dtlz2_front(4, 6)
    indicator = mi.HVApproxIndicator(reference_point=np.full(4, 1.1))

    first = indicator.compute(front, seed=1, method="DZ2019-MC")
    again = indicator.compute(front, seed=1, method="DZ2019-MC")
    other = indicator.compute(front, seed=2, method="DZ2019-MC")

    assert first.value == again.value
    assert other.value != first.value
    assert (first.details["seed"], first.details["method"]) == (1, "DZ2019-MC")


@pytest.mark.skipif(not mi.has_moocore(), reason="MooCore not installed")
def test_hv_approx_records_moocores_default_method() -> None:
    import inspect

    import moocore

    front = _dtlz2_front(3, 4)
    ref = np.full(3, 1.1)

    result = mi.HVApproxIndicator(reference_point=ref).compute(front)

    default = inspect.signature(moocore.hv_approx).parameters["method"].default
    assert {key: result.details[key] for key in ("samples", "seed", "method")} == {"samples": 10_000, "seed": None, "method": default}
    np.testing.assert_array_equal(result.details["reference_point"], ref)
    assert result.value == moocore.hv_approx(front, ref=ref, nsamples=10_000, method=default)


@pytest.mark.skipif(not mi.has_moocore(), reason="MooCore not installed")
def test_hv_approx_rejects_an_unknown_method() -> None:
    with pytest.raises(ValueError, match=r"'Rphi-FWE\+', 'DZ2019-HW', 'DZ2019-MC'"):
        mi.HVApproxIndicator(reference_point=np.full(2, 1.1)).compute(_REFERENCE_FRONT, method="monte-carlo")


def test_the_analysis_facade_exports_the_exact_and_the_approximate_hypervolume() -> None:
    from vamos.ux import api

    assert api.HVApproxIndicator is mi.HVApproxIndicator is type(mi.get_indicator("hv_approx"))
    assert api.HVIndicator is mi.HVIndicator
    assert api.IndicatorResult is mi.IndicatorResult
    assert api.HV_APPROX_METHODS == mi.HV_APPROX_METHODS
