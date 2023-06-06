"""Hellinger distance module."""

from typing import List, Optional, Union

import numpy as np  # type: ignore

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.batch.distance_based.base import (
    BaseDistanceBasedBins,
)


class HellingerDistance(BaseDistanceBasedBins):
    """Hellinger distance [hellinger1909neue]_ detector.

    :References:

    .. [hellinger1909neue] Hellinger, Ernst.
        "Neue begründung der theorie quadratischer formen von unendlichvielen
        veränderlichen."
        Journal für die reine und angewandte Mathematik 1909.136 (1909): 210-271.
    """

    def __init__(
        self,
        num_bins: int = 10,
        callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]] = None,
    ) -> None:
        """Init method.

        :param num_bins: number of bins in which to divide probabilities
        :type num_bins: int
        :param callbacks: callbacks
        :type callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]]
        """
        sqrt_div = np.sqrt(2)
        super().__init__(
            statistical_method=self._hellinger,
            statistical_kwargs={
                "num_bins": num_bins,
                "sqrt_div": sqrt_div,
            },
            callbacks=callbacks,
        )
        self.num_bins = num_bins
        self.sqrt_div = sqrt_div

    def _distance_measure_bins(
        self,
        X_ref: np.ndarray,  # noqa: N803
        X: np.ndarray,  # noqa: N803
    ) -> float:
        hellinger = self._hellinger(
            X=X_ref,
            Y=X,
            num_bins=self.num_bins,
            sqrt_div=self.sqrt_div,
        )
        return hellinger

    @staticmethod
    def _hellinger(
        X: np.ndarray, Y: np.ndarray, *, num_bins: int, sqrt_div: float  # noqa: N803
    ) -> float:
        (  # noqa: N806
            X_percents,
            Y_percents,
        ) = BaseDistanceBasedBins._calculate_bins_values(
            X_ref=X, X=Y, num_bins=num_bins
        )
        hellinger = (
            np.sqrt(np.sum((np.sqrt(X_percents) - np.sqrt(Y_percents)) ** 2)) / sqrt_div
        )
        return hellinger
