"""Kernels module."""

import numpy as np  # type: ignore
from scipy.spatial.distance import cdist  # type: ignore


def rbf_kernel(
    X: np.ndarray, Y: np.ndarray, sigma: float = 1.0  # noqa: N803
) -> np.ndarray:
    """Radial basis function kernel between X and Y matrices.

    :param X: X matrix
    :type X: numpy.ndarray
    :param Y: Y matrix
    :type Y: numpy.ndarray
    :param sigma: sigma value (equivalent to gamma = 1 / (2 * sigma**2))
    :type sigma: float
    :return: Radial basis kernel matrix
    :rtype: numpy.ndarray
    """
    return np.exp(-cdist(X, Y, "sqeuclidean") / (2 * sigma**2))
