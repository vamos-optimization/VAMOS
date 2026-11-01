import numpy as np
import pytest

import vamos.foundation.quality_indicators.hypervolume as hv


def test_hypervolume_impl_2d_matches_known_value():
    points = np.array([[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]], dtype=float)
    ref = np.array([1.0, 1.0], dtype=float)
    assert hv._hypervolume_impl(points, ref) == pytest.approx(0.37)


def test_hypervolume_impl_2d_ignores_dominated_points():
    points = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=float)
    ref = np.array([3.0, 3.0], dtype=float)
    assert hv._hypervolume_impl(points, ref) == pytest.approx(4.0)


def test_hypervolume_contributions_2d_matches_bruteforce():
    points = np.array(
        [
            [0.2, 0.8],
            [0.8, 0.2],
            [0.5, 0.9],  # dominated by [0.2, 0.8]
            [0.2, 0.8],  # duplicate of the first point
        ],
        dtype=float,
    )
    ref = np.array([1.0, 1.0], dtype=float)
    hv_full = hv._hypervolume_impl(points, ref)
    expected = np.array(
        [hv_full - hv._hypervolume_impl(np.delete(points, i, axis=0), ref) for i in range(points.shape[0])],
        dtype=float,
    )
    assert hv._hypervolume_contributions_2d(points, ref) == pytest.approx(expected)


def test_hypervolume_filters_points_outside_the_reference() -> None:
    points = np.array([[0.2, 1.2], [0.5, 0.5]], dtype=float)
    ref = np.array([1.0, 1.0], dtype=float)

    assert hv.hypervolume(points, ref) == pytest.approx(0.25)
    assert hv.hypervolume_contributions(points, ref) == pytest.approx([0.0, 0.25])
    with pytest.warns(FutureWarning, match="allow_ref_expand"):
        assert hv.hypervolume(points, ref, allow_ref_expand=True) == pytest.approx(0.25)
    with pytest.warns(FutureWarning, match="allow_ref_expand"):
        assert hv.hypervolume_contributions(points, ref, allow_ref_expand=True) == pytest.approx([0.0, 0.25])


def _without_optional_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hv, "_OPTIONAL_LOADED", True)
    monkeypatch.setattr(hv, "_MOOCORE", None)
    monkeypatch.setattr(hv, "_LIBHV_MODULE", None)
    monkeypatch.setattr(hv, "_LIBHV_HV_FN", None)
    monkeypatch.setattr(hv, "_LIBHV_CLASS", None)


@pytest.mark.parametrize("n_obj", [3, 4])
def test_fallback_contributions_match_moocore_with_dominated_points(n_obj: int, monkeypatch: pytest.MonkeyPatch) -> None:
    moocore = pytest.importorskip("moocore")
    rng = np.random.default_rng(7)
    front = rng.random((10, n_obj))
    points = np.vstack([front, front[:3] + 0.05, front[4]])  # three dominated points and a duplicate
    ref = np.full(n_obj, 1.1)
    expected = np.asarray(moocore.hv_contributions(points, ref=ref), dtype=float)

    _without_optional_backends(monkeypatch)
    actual = hv.hypervolume_contributions(points, ref)

    assert actual == pytest.approx(expected, abs=1e-12)
    assert (actual >= 0.0).all()
    outside = np.vstack([points, np.full((1, n_obj), 0.5) + np.eye(n_obj)[0]])
    assert hv.hypervolume_contributions(outside, ref) == pytest.approx(np.append(expected, 0.0), abs=1e-12)
    assert hv.hypervolume(outside, ref) == pytest.approx(moocore.hypervolume(points, ref=ref), abs=1e-12)


def test_hypervolume_rejects_non_finite_points_and_reference():
    ref = np.array([1.0, 1.0], dtype=float)

    with pytest.raises(ValueError, match="finite"):
        hv.hypervolume(np.array([[np.nan, 0.2]], dtype=float), ref)
    with pytest.raises(ValueError, match="finite"):
        hv.hypervolume_contributions(np.array([[0.2, 0.2]], dtype=float), np.array([np.inf, 1.0]))


def test_hypervolume_supports_three_objective_fronts():
    points = np.array([[1.0, 1.0, 1.0]], dtype=float)
    ref = np.array([2.0, 2.0, 2.0], dtype=float)

    assert hv.hypervolume(points, ref) == pytest.approx(1.0)
    assert hv.hypervolume_contributions(points, ref) == pytest.approx(np.array([1.0]))
