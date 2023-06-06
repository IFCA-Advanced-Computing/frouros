"""HI (Histogram intersection) normalized complement module."""

from typing import List, Optional, Union

import numpy as np  # type: ignore

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.batch.distance_based.base import (
    BaseDistanceBasedBins,
)


class HINormalizedComplement(BaseDistanceBasedBins):
    """HI (Histogram intersection) normalized complement [swain1991color]_ detector.

    :References:

    .. [swain1991color] Swain, M. J., and D. H. Ballard.
        "Color Indexing International Journal of Computer
        Vision 7." (1991): 11-32.
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
        super().__init__(
            statistical_method=self._hi_normalized_complement,
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
        intersection_normalized_complement = self._hi_normalized_complement(
            X=X_ref, Y=X, num_bins=self.num_bins
        )
        return intersection_normalized_complement

    @staticmethod
    def _hi_normalized_complement(
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
        intersection_normalized_complement = 1 - np.sum(np.minimum(X_hist, Y_hist))

        return intersection_normalized_complement
