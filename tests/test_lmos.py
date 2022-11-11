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
