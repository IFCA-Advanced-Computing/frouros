"""Bhattacharyya distance module."""

from typing import Optional, Union

import numpy as np

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.base import MultivariateData
from frouros.detectors.data_drift.batch.distance_based.base import (
    BaseDistanceBasedBins,
)


class BhattacharyyaDistance(BaseDistanceBasedBins):
    """Bhattacharyya distance [bhattacharyya1946measure]_ detector.

    :param num_bins: number of bins per dimension in which to
    divide probabilities, defaults to 10
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
    >>> X = np.random.multivariate_normal(mean=[1, 1], cov=[[2, 0], [0, 2]], size=100)
    >>> Y = np.random.multivariate_normal(mean=[0, 0], cov=[[2, 1], [1, 2]], size=100)
    >>> detector = BhattacharyyaDistance(num_bins=10)
    >>> _ = detector.fit(X=X)
    >>> detector.compare(X=Y)
    DistanceResult(distance=0.3413868461814531)
    """

    def __init__(  # noqa: D107
        self,
        num_bins: int = 10,
        callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]] = None,
    ) -> None:
        super().__init__(
            statistical_type=MultivariateData(),
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
        bhattacharyya = self._bhattacharyya(
            X=X_ref,
            Y=X,
            num_bins=self.num_bins,
        )
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
            X_ref=X,
            X=Y,
            num_bins=num_bins,
        )

        # Add small epsilon to avoid log(0)
        epsilon = np.finfo(float).eps
        X_percents = X_percents + epsilon
        Y_percents = Y_percents + epsilon

        # Compute Bhattacharyya coefficient
        bc = np.sum(np.sqrt(X_percents * Y_percents))
        # Clip between [0,1] to avoid numerical errors
        bc = np.clip(bc, a_min=0, a_max=1)

        # Compute Bhattacharyya distance
        # Use absolute value to avoid negative zero values
        bhattacharyya = np.abs(-np.log(bc))

        return bhattacharyya
