import numpy as np
import pytest

import lmos


def test_median_of_squares():
    """Tests for lmos.median_of_squares"""
    A_even = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    A_odd = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    rhs_even = np.array([4, 9, 30, 15])
    rhs_odd = np.array([3, 11, 20, 23, 42])
    theta = np.array([5, -1])

    assert lmos.median_of_squares(theta, A_even, rhs_even) == 62.5
    assert lmos.median_of_squares(theta, A_odd, rhs_odd) == 1.0
    with pytest.raises(ValueError):
        lmos.median_of_squares(theta, A_even, rhs_odd)
    with pytest.raises(ValueError):
        lmos.median_of_squares(theta, A_odd, rhs_even)
    with pytest.raises(ValueError):
        lmos.median_of_squares(np.array([1, 2, 3]), A_odd, rhs_odd)


def test_numerically_optimize():
    """Tests for lmos.numerically_optimize"""
    np.random.seed(0)
    n = 201
    A = np.ones((n, 2))
    A[:, 1] = np.arange(n)
    rhs = 2 * A[:, 1] + 3 + np.random.normal(0, 30, n)
    initial_theta = np.array([10, 5])
    theta, error = lmos.numerically_optimize(initial_theta, A, rhs)
    assert 470 < error < 480
    assert 1.9 < theta[1] < 2.1
    assert error == lmos.median_of_squares(theta, A, rhs)
