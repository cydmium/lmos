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


def numerically_optimize(
    initial_guess: np.ndarray, A: np.ndarray, rhs: np.ndarray
) -> tuple[np.ndarray, np.float64]:
    """Attempt to compute the optimal theta using scipy's minimize functionality

    Parameters
    ----------
    initial_guess: np.ndarray
        Intial guess for theta
    A: np.ndarray
        Feature array
    rhs: np.ndarray
        Vector for the right hand side, i.e. what A * theta should be

    Returns
    -------
    tuple[np.array, mos]:
        Optimized theta vector and median of squared residuals
    """
    res = scipy.optimize.minimize(median_of_squares, initial_guess, (A, rhs))
    return (res.x, res.fun)
