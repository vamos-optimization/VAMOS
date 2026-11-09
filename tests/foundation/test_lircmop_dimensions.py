"""LIR-CMOP's dimensions: LIR-CMOP1-4 need a variable in each of their two distance groups."""

from __future__ import annotations

import numpy as np
import pytest

from vamos.foundation.problem.lircmop import LIRCMOPProblem


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_lircmop1_to_4_reject_an_empty_distance_group(k: int) -> None:
    # With two variables the third, where g1 starts, is missing: g1 = 0 and no design meets g1 in [0.5, 0.51].
    with pytest.raises(ValueError, match="at least 3 variables"):
        LIRCMOPProblem(k, n_var=2)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_lircmop1_to_4_with_three_variables_have_feasible_designs(k: int) -> None:
    # x1 = 0.025 meets sin(20 pi x1) >= 0.5 (LIR-CMOP3 and 4); the other two put g1 and g2 at 0.5.
    x1 = 0.025
    x2 = np.cos(0.5 * np.pi * x1) - np.sqrt(0.5)
    x3 = np.sin(0.5 * np.pi * x1) + np.sqrt(0.5)
    out: dict[str, np.ndarray] = {}

    LIRCMOPProblem(k, n_var=3).evaluate(np.array([[x1, x2, x3]]), out)

    assert np.all(out["G"] <= 1e-12)


@pytest.mark.parametrize("k", [5, 9, 12])
def test_other_lircmops_accept_as_many_variables_as_objectives(k: int) -> None:
    assert LIRCMOPProblem(k, n_var=2).n_var == 2
