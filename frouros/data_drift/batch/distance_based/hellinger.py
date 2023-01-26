"""Hellinger distance module."""

import numpy as np  # type: ignore

from frouros.data_drift.batch.distance_based.base import (
    DistanceBinsBasedBase,
)


class Hellinger(DistanceBinsBasedBase):
    """Hellinger algorithm class."""

    def __init__(self, num_bins: int = 10) -> None:
        """Init method.

        :param num_bins: number of bins in which to divide probabilities
        :type num_bins: int
        """
        super().__init__(num_bins=num_bins)
        self._sqrt_div = np.sqrt(2)

    def _distance_measure_bins(
        self,
        X_ref_: np.ndarray,  # noqa: N803
        X: np.ndarray,  # noqa: N803
    ) -> float:
        distance = self._hellinger(
            X_ref_=X_ref_,
            X=X,
            sqrt_div=self._sqrt_div,
        )
        return distance

    @staticmethod
    def _hellinger(
        X_ref_: np.ndarray, X: np.ndarray, sqrt_div: float  # noqa: N803
    ) -> float:
        distance = np.sqrt(np.sum((np.sqrt(X_ref_) - np.sqrt(X)) ** 2)) / sqrt_div
        return distance
