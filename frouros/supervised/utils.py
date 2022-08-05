"""Supervised util functions."""

from typing import List, Tuple

import numpy as np  # type: ignore


def list_to_arrays(list_: List[Tuple[np.ndarray, np.ndarray]]):
    """Convert list to numpy arrays.

    :param list_: list of samples
    :type list_: List[Tuple[numpy.ndarray, numpy.ndarray]]
    :return: list of numpy arrays
    :rtype List[numpy.ndarray]
    """
    X, y = [*map(np.array, zip(*list_))]  # noqa: N806
    X = np.squeeze(X, axis=1)  # noqa: N806
    y = np.ravel(y)
    return X, y
