import numpy as np

from vamos.foundation.constraints import compute_constraint_info


def test_compute_constraint_info_basic():
    # Arrange
    G = np.array([[-1.0, 0.0], [0.2, -0.1], [0.5, 0.5]])

    # Act
    info = compute_constraint_info(G)

    # Assert
    assert np.allclose(info.cv, [0.0, 0.2, 1.0])
    assert info.feasible_mask.tolist() == [True, False, False]


def test_compute_constraint_info_respects_eps():
    G = np.array([[0.05], [0.1], [0.15]])

    info = compute_constraint_info(G, eps=0.1)

    assert info.feasible_mask.tolist() == [True, True, False]
    assert np.allclose(info.cv, [0.0, 0.0, 0.05])
