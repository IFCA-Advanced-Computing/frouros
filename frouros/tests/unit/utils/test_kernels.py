"""Test kernels module."""

import numpy as np
import pytest

from frouros.utils.kernels import rbf_kernel

# TODO: Create fixtures for the matrices and the expected kernel values


@pytest.mark.parametrize(
    "X, Y, sigma, expected_kernel_value",
    [
        (np.array([[1, 2, 3]]), np.array([[1, 2, 3]]), 0.5, np.array([[1.0]])),
        (np.array([[1, 2, 3]]), np.array([[1, 2, 3]]), 1.0, np.array([[1.0]])),
        (np.array([[1, 2, 3]]), np.array([[1, 2, 3]]), 2.0, np.array([[1.0]])),
        (
            np.array([[1, 2, 3]]),
            np.array([[4, 5, 6]]),
            0.5,
            np.array([[3.53262857e-24]]),
        ),
        (
            np.array([[1, 2, 3]]),
            np.array([[4, 5, 6]]),
            1.0,
            np.array([[1.37095909e-06]]),
        ),
        (np.array([[1, 2, 3]]), np.array([[4, 5, 6]]), 2.0, np.array([[0.03421812]])),
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[1, 2, 3], [4, 5, 6]]),
            0.5,
            np.array(
                [[1.00000000e00, 3.53262857e-24], [3.53262857e-24, 1.00000000e00]]
            ),
        ),
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[1, 2, 3], [4, 5, 6]]),
            1.0,
            np.array(
                [[1.00000000e00, 1.37095909e-06], [1.37095909e-06, 1.00000000e00]]
            ),
        ),
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[1, 2, 3], [4, 5, 6]]),
            2.0,
            np.array([[1.00000000e00, 0.03421812], [0.03421812, 1.00000000e00]]),
        ),
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]),
            0.5,
            np.array(
                [[2.23130160e-01, 1.20048180e-32], [5.17555501e-17, 2.23130160e-01]]
            ),
        ),
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]),
            1.0,
            np.array(
                [[6.87289279e-01, 1.04674018e-08], [8.48182352e-05, 6.87289279e-01]]
            ),
        ),
        (
            np.array([[1, 2, 3], [4, 5, 6]]),
            np.array([[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]),
            2.0,
            np.array([[0.91051036, 0.01011486], [0.09596709, 0.91051036]]),
        ),
    ],
)
def test_rbf_kernel(
    X: np.ndarray,  # noqa: N803
    Y: np.ndarray,
    sigma: float,
    expected_kernel_value: np.ndarray,
) -> None:
    """Test rbf kernel.

    :param X: X values
    :type X: numpy.ndarray
    :param Y: Y values
    :type Y: numpy.ndarray
    :param sigma: sigma value
    :type sigma: float
    :param expected_kernel_value: expected kernel value
    :type expected_kernel_value: numpy.ndarray
    """
    assert np.all(
        np.isclose(
            rbf_kernel(
                X=X,
                Y=Y,
                sigma=sigma,
            ),
            expected_kernel_value,
        ),
    )
