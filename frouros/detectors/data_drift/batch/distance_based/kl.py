"""KL (Kullback-Leibler divergence distance) module."""

from typing import Any, Dict, List, Optional, Union

import numpy as np  # type: ignore
from scipy.special import rel_entr  # type: ignore

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.batch.distance_based.base import (
    BaseDistanceBasedProbability,
    DistanceResult,
)


class KL(BaseDistanceBasedProbability):
    """KL (Kullback-Leibler divergence) [kullback1951information]_ detector.

    :References:

    .. [kullback1951information] Kullback, Solomon, and Richard A. Leibler.
        "On information and sufficiency."
        The annals of mathematical statistics 22.1 (1951): 79-86.
    """

    def __init__(
        self,
        num_bins: int = 10,
        callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]] = None,
        **kwargs,
    ) -> None:
        """Init method.

        :param num_bins: number of bins in which to divide probabilities
        :type num_bins: int
        :param callbacks: number of bins in which to divide probabilities
        :type callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]]
        """
        super().__init__(
            statistical_method=self._kl,
            statistical_kwargs={**kwargs, "num_bins": num_bins},
            callbacks=callbacks,
        )
        self.num_bins = num_bins
        self.kwargs = kwargs

    def _distance_measure(
        self,
        X_ref: np.ndarray,  # noqa: N803
        X: np.ndarray,  # noqa: N803
        **kwargs,
    ) -> DistanceResult:
        kl = self._kl(X=X_ref, Y=X, num_bins=self.num_bins, **self.kwargs)
        distance = DistanceResult(distance=kl)
        return distance

    @staticmethod
    def _kl(
        X: np.ndarray,  # noqa: N803
        Y: np.ndarray,
        *,
        num_bins: int,
        **kwargs: Dict[str, Any],
    ) -> float:
        (  # noqa: N806
            X_ref_rvs,
            X_rvs,
        ) = BaseDistanceBasedProbability._calculate_probabilities(
            X_ref=X,
            X=Y,
            num_bins=num_bins,
        )
        kl = np.sum(rel_entr(X_rvs, X_ref_rvs, **kwargs))
        return kl
