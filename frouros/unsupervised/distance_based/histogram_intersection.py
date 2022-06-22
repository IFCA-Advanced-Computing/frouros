"""Histogram intersection module."""

import numpy as np  # type: ignore

from frouros.unsupervised.base import UnivariateTest
from frouros.unsupervised.distance_based.base import (  # type: ignore
    DistanceBasedEstimator,
)


class HistogramIntersection(DistanceBasedEstimator):
    """Histogram intersection algorithm class."""

    def __init__(self, num_bins: int = 100) -> None:
        """Init method."""
        super().__init__(test_type=UnivariateTest())
        self.num_bins = num_bins

    @staticmethod
    def _histogram_intersection(
        X_ref_hist: np.ndarray, X_hist: np.ndarray  # noqa: N803
    ) -> float:
        intersection = np.sum(np.minimum(X_ref_hist, X_hist))
        return intersection

    def _distance(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> float:
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
        distance = 1 - self._histogram_intersection(
            X_ref_hist=X_ref_hist, X_hist=X_hist
        )
        return distance
