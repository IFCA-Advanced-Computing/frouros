"""Unsupervised distance based base module."""

import abc
from typing import Tuple, Union

import numpy as np  # type: ignore

from frouros.unsupervised.base import UnsupervisedBaseEstimator


class DistanceBasedEstimator(UnsupervisedBaseEstimator):
    """Abstract class representing a distance based estimator."""

    def _apply_method(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Union[Tuple[float, float], np.float]:
        distance = self._distance(X_ref_=X_ref_, X=X, **kwargs)
        return distance

    @abc.abstractmethod
    def _distance(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Union[Tuple[np.float, np.float], np.float]:
        pass
