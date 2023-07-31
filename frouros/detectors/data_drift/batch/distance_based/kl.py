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
    """KL (Kullback-Leibler divergence) [1]_ detector.

    :param num_bins: number of bins in which to divide probabilities, defaults to 10
    :type num_bins: int
    :param callbacks: number of bins in which to divide probabilities, defaults to None
    :type callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]]
    :param kwargs: additional keyword arguments to pass to scipy.special.rel_entr
    :type kwargs: Dict[str, Any]

    :References:

    .. [1] Kullback, Solomon, and Richard A. Leibler.
        "On information and sufficiency."
        The annals of mathematical statistics 22.1 (1951): 79-86.

    :Example:

    >>> from frouros.detectors.data_drift import KL
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> X = np.random.normal(loc=0, scale=1, size=100)
    >>> Y = np.random.normal(loc=1, scale=1, size=100)
    >>> detector = KL(num_bins=20)
    >>> _ = detector.fit(X=X)
    >>> result, _ = detector.compare(X=Y)
    """

    def __init__(
        self,
        num_bins: int = 10,
        callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]] = None,
        **kwargs,
    ) -> None:
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
