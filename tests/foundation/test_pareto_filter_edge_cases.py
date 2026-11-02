from __future__ import annotations

import numpy as np
import pytest

from vamos.foundation.quality_indicators.pareto import pareto_filter


def test_pareto_filter_keeps_duplicates() -> None:
    F = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    front, idx = pareto_filter(F, return_indices=True)

    assert front.shape[0] == 2
    assert set(idx.tolist()) == {0, 1}


def test_pareto_filter_handles_none() -> None:
    front, idx = pareto_filter(None, return_indices=True)
    assert front.shape == (0, 0)
    assert idx.size == 0


def test_pareto_filter_rejects_nan_rows_and_one_dimensional_input() -> None:
    with pytest.raises(ValueError, match="NaN"):
        pareto_filter(np.array([[np.nan, 0.0], [1.0, 1.0]]))
    with pytest.raises(ValueError, match="2D"):
        pareto_filter(np.array([1.0, 2.0, 3.0]))
    assert pareto_filter(np.array([[np.inf, 0.0], [1.0, 1.0]])).tolist() == [[np.inf, 0.0], [1.0, 1.0]]
