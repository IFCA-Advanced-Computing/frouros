"""Hellinger distance module."""

from typing import Optional, Union

import numpy as np

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.base import UnivariateData
from frouros.detectors.data_drift.batch.distance_based.base import (
    BaseDistanceBasedBins,
)


class HellingerDistance(BaseDistanceBasedBins):
    """Hellinger distance [hellinger1909neue]_ detector.

    :param num_bins: number of bins in which to divide probabilities, defaults to 10
    :type num_bins: int
    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]]

    :References:

    .. [hellinger1909neue] Hellinger, Ernst.
        "Neue begründung der theorie quadratischer formen von unendlichvielen
        veränderlichen."
        Journal für die reine und angewandte Mathematik 1909.136 (1909): 210-271.

    :Example:

    >>> from frouros.detectors.data_drift import HellingerDistance
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> X = np.random.normal(loc=0, scale=1, size=100)
    >>> Y = np.random.normal(loc=1, scale=1, size=100)
    >>> detector = HellingerDistance(num_bins=20)
    >>> _ = detector.fit(X=X)
    >>> detector.compare(X=Y)[0]
    DistanceResult(distance=0.467129645775421)
    """

    def __init__(  # noqa: D107
        self,
        num_bins: int = 10,
        callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]] = None,
    ) -> None:
        sqrt_div = np.sqrt(2)
        super().__init__(
            statistical_type=UnivariateData(),
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
        X: np.ndarray,  # noqa: N803
        Y: np.ndarray,
        *,
        num_bins: int,
        sqrt_div: float,
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
