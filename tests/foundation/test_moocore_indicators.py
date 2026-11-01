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
