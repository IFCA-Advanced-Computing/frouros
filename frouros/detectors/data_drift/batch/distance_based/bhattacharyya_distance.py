"""Bhattacharyya distance module."""

from typing import Optional, Union

import numpy as np

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.batch.distance_based.base import (
    BaseDistanceBasedBins,
)


class BhattacharyyaDistance(BaseDistanceBasedBins):
    """Bhattacharyya distance [bhattacharyya1946measure]_ detector.

    :param num_bins: number of bins in which to divide probabilities, defaults to 10
    :type num_bins: int
    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallback, list[Callback]]]

    :References:

    .. [bhattacharyya1946measure] Bhattacharyya, Anil.
        "On a measure of divergence between two multinomial populations."
        Sankhyā: the indian journal of statistics (1946): 401-406.

    :Example:

    >>> from frouros.detectors.data_drift import BhattacharyyaDistance
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> X = np.random.normal(loc=0, scale=1, size=100)
    >>> Y = np.random.normal(loc=1, scale=1, size=100)
    >>> detector = BhattacharyyaDistance(num_bins=20)
    >>> _ = detector.fit(X=X)
    >>> detector.compare(X=Y)
    DistanceResult(distance=0.2182101059622703)
    """

    def __init__(  # noqa: D107
        self,
        num_bins: int = 10,
        callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]] = None,
    ) -> None:
        super().__init__(
            statistical_method=self._bhattacharyya,
            statistical_kwargs={
                "num_bins": num_bins,
            },
            callbacks=callbacks,
        )
        self.num_bins = num_bins

    def _distance_measure_bins(
        self,
        X_ref: np.ndarray,  # noqa: N803
        X: np.ndarray,  # noqa: N803
    ) -> float:
        bhattacharyya = self._bhattacharyya(X=X_ref, Y=X, num_bins=self.num_bins)
        return bhattacharyya

    @staticmethod
    def _bhattacharyya(
        X: np.ndarray,  # noqa: N803
        Y: np.ndarray,
        *,
        num_bins: int,
    ) -> float:
        (  # noqa: N806
            X_percents,
            Y_percents,
        ) = BaseDistanceBasedBins._calculate_bins_values(
            X_ref=X, X=Y, num_bins=num_bins
        )
        bhattacharyya = 1 - np.sum(np.sqrt(X_percents * Y_percents))
        return bhattacharyya
