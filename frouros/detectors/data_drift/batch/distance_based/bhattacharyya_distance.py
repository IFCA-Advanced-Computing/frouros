"""Bhattacharyya distance module."""

from typing import List, Optional, Union

import numpy as np  # type: ignore

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.batch.distance_based.base import (
    BaseDistanceBasedBins,
)


class BhattacharyyaDistance(BaseDistanceBasedBins):
    """Bhattacharyya distance [bhattacharyya1946measure]_ detector.

    :References:

    .. [bhattacharyya1946measure] Bhattacharyya, Anil.
        "On a measure of divergence between two multinomial populations."
        Sankhyā: the indian journal of statistics (1946): 401-406.
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
        :type callbacks: Optional[Union[BaseCallback, List[Callback]]]
        """
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
        X: np.ndarray, Y: np.ndarray, *, num_bins: int  # noqa: N803
    ) -> float:
        (  # noqa: N806
            X_percents,
            Y_percents,
        ) = BaseDistanceBasedBins._calculate_bins_values(
            X_ref=X, X=Y, num_bins=num_bins
        )
        bhattacharyya = 1 - np.sum(np.sqrt(X_percents * Y_percents))
        return bhattacharyya
