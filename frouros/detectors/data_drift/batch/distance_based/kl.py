"""KL (Kullback-Leibler divergence distance) module."""

from typing import Any, Optional, Union

import numpy as np
from scipy.special import rel_entr

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.batch.distance_based.base import (
    BaseDistanceBasedProbability,
    DistanceResult,
)


class KL(BaseDistanceBasedProbability):
    """KL (Kullback-Leibler divergence) [kullback1951information]_ detector.

    :param num_bins: number of bins in which to divide probabilities, defaults to 10
    :type num_bins: int
    :param callbacks: number of bins in which to divide probabilities, defaults to None
    :type callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]]
    :param kwargs: additional keyword arguments to pass to scipy.special.rel_entr
    :type kwargs: dict[str, Any]

    :References:

    .. [kullback1951information] Kullback, Solomon, and Richard A. Leibler.
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
    >>> detector.compare(X=Y)[0]
    DistanceResult(distance=inf)
    """

    def __init__(  # noqa: D107
        self,
        num_bins: int = 10,
        callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]] = None,
        **kwargs: Any,
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
        **kwargs: Any,
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
        **kwargs: dict[str, Any],
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
