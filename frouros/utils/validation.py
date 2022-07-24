"""Validation module."""

import numpy as np  # type: ignore

from frouros.common.exceptions import OneSampleError


def check_is_one_sample(array: np.ndarray) -> None:
    """Check that a numpy array has only one value.

    :param array: numpy array
    :type array: numpy.ndarray
    :raise TypeError: Type error exception
    :raise SamplesError: Samples error exception
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("y must be of type numpy.ndarray")
    if array.shape[0] != 1:
        raise OneSampleError(
            f"Only one sample at a time is supported but {array.shape[0]} were found."
        )
