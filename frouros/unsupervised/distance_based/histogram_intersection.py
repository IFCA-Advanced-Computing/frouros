"""Histogram intersection module."""

import numpy as np  # type: ignore

from frouros.unsupervised.base import NumericalData, UnivariateType
from frouros.unsupervised.distance_based.base import (
    DistanceBasedEstimator,
    DistanceResult,
)


class HistogramIntersection(DistanceBasedEstimator):
    """Histogram intersection algorithm class."""

    def __init__(self, num_bins: int = 100) -> None:
        """Init method.

        :param num_bins: number of bins in which to divide probabilities
        :type num_bins: int
        """
        super().__init__(data_type=NumericalData(), statistical_type=UnivariateType())
        self.num_bins = num_bins

    @property
    def num_bins(self) -> int:
        """Number of bins property.

        :return: number of bins in which to divide probabilities
        :rtype: int
        """
        return self._num_bins

    @num_bins.setter
    def num_bins(self, value: int) -> None:
        """Number of bins setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 1:
            raise ValueError("value must be greater than 0.")
        self._num_bins = value

    @staticmethod
    def _histogram_intersection(
        X_ref_hist: np.ndarray, X_hist: np.ndarray  # noqa: N803
    ) -> float:
        intersection = np.sum(np.minimum(X_ref_hist, X_hist))
        return intersection

    def _distance_measure(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> DistanceResult:
        hist_range = (
            np.min([np.min(X_ref_), np.min(X)]),
            np.max([np.max(X_ref_), np.max(X)]),
        )
        X_ref_hist, _ = np.histogram(  # noqa: N806
            X_ref_, bins=self.num_bins, range=hist_range  # noqa: N806
        )
        X_ref_hist = X_ref_hist / X_ref_.shape[0]  # noqa: N806
        X_hist, _ = np.histogram(X, bins=self.num_bins, range=hist_range)  # noqa: N806
        X_hist = X_hist / X.shape[0]  # noqa: N806
        intersection = 1 - self._histogram_intersection(
            X_ref_hist=X_ref_hist, X_hist=X_hist
        )
        distance = DistanceResult(distance=intersection)
        return distance
