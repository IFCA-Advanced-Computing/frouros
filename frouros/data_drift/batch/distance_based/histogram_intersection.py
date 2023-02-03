"""Histogram intersection module."""

from typing import List, Optional, Union

import numpy as np  # type: ignore

from frouros.callbacks import Callback
from frouros.data_drift.batch.distance_based.base import (
    DistanceBinsBasedBase,
)


class HistogramIntersection(DistanceBinsBasedBase):
    """Histogram intersection algorithm class."""

    def __init__(
        self,
        num_bins: int = 10,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
    ) -> None:
        """Init method.

        :param num_bins: number of bins in which to divide probabilities
        :type num_bins: int
        :param callbacks: callbacks
        :type callbacks: Optional[Union[Callback, List[Callback]]]
        """
        super().__init__(
            statistical_method=self._histogram_intersection,
            statistical_kwargs={"num_bins": num_bins},
            callbacks=callbacks,
        )
        self.num_bins = num_bins

    def _distance_measure_bins(
        self,
        X_ref_: np.ndarray,  # noqa: N803
        X: np.ndarray,  # noqa: N803
    ) -> float:
        no_intersection = self._histogram_intersection(
            X=X_ref_, Y=X, num_bins=self.num_bins
        )
        return no_intersection

    @staticmethod
    def _histogram_intersection(
        X: np.ndarray,  # noqa: N803
        Y: np.ndarray,
        *,
        num_bins: int,
    ) -> float:
        hist_range = (
            np.min([np.min(X), np.min(Y)]),
            np.max([np.max(X), np.max(Y)]),
        )
        X_hist, _ = np.histogram(  # noqa: N806
            X, bins=num_bins, range=hist_range  # noqa: N806
        )
        X_hist = X_hist / X.shape[0]  # noqa: N806
        Y_hist, _ = np.histogram(Y, bins=num_bins, range=hist_range)  # noqa: N806
        Y_hist = Y_hist / Y.shape[0]  # noqa: N806
        no_intersection = 1 - np.sum(np.minimum(X_hist, Y_hist))

        return no_intersection
