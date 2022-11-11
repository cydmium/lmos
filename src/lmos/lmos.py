import numpy as np


def median_of_squares(theta: np.ndarray, A: np.ndarray, rhs: np.ndarray) -> np.float64:
    """Compute the median of squared residuals

    Parameters
    ----------
    theta: np.ndarray
        Current solution to be tested
    A: np.ndarray
        Feature array
    rhs: np.ndarray
        Vector for the right hand side, i.e. what A * theta should be

    Returns
    -------
    np.float64:
        The median of the squared residuals
    """
    return np.median(np.power(np.abs(A @ theta - rhs), 2))
