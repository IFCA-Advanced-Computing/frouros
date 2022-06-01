"""Numerical detectors module."""

import numpy as np  # type: ignore

from frouros.transformations.detectors.base import BaseDetectors


class NumericalDetectors(BaseDetectors):
    """Numerical detectors transformation."""

    def preprocess_x(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        """Preprocess X method.

        :param X: feature data
        :type X: numpy.ndarray
        :return: preprocessed feature data
        :rtype: numpy.ndarray
        """
        return X
