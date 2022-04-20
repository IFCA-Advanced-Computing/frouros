"""Unsupervised distance based base module."""

import abc

import numpy as np  # type: ignore

from frouros.unsupervised.base import UnsupervisedBaseEstimator


class DistanceBasedEstimator(UnsupervisedBaseEstimator):
    """Abstract class representing a distance based estimator."""

    def _apply_method(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> float:
        distance = self._distance(X_ref_=X_ref_, X=X, **kwargs)
        return distance

    @staticmethod
    @abc.abstractmethod
    def _distance(X_ref_: np.ndarray, X: np.ndarray, **kwargs) -> float:  # noqa: N803
        pass
