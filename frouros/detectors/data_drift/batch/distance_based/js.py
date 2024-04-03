"""JS (Jensen-Shannon distance) module."""

from typing import Any, Optional, Union

import numpy as np
from scipy.spatial.distance import jensenshannon

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.batch.distance_based.base import (
    BaseDistanceBasedProbability,
    DistanceResult,
)


class JS(BaseDistanceBasedProbability):
    """JS (Jensen-Shannon distance) [lin1991divergence]_ detector.

    :param num_bins: number of bins in which to divide probabilities, defaults to 10
    :type num_bins: int
    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]]
    :param kwargs: additional keyword arguments to pass to scipy.spatial.distance.jensenshannon
    :type kwargs: dict[str, Any]

    :References:

    .. [lin1991divergence] Lin, Jianhua.
        "Divergence measures based on the Shannon entropy."
        IEEE Transactions on Information theory 37.1 (1991): 145-151.

    :Example:

    >>> from frouros.detectors.data_drift import JS
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> X = np.random.normal(loc=0, scale=1, size=100)
    >>> Y = np.random.normal(loc=1, scale=1, size=100)
    >>> detector = JS(num_bins=20)
    >>> _ = detector.fit(X=X)
    >>> detector.compare(X=Y)[0]
    DistanceResult(distance=0.41702877367162156)
    """  # noqa: E501

    def __init__(  # noqa: D107
        self,
        num_bins: int = 10,
        callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            statistical_method=self._js,
            statistical_kwargs={
                "num_bins": num_bins,
                **kwargs,
            },
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
        js = self._js(X=X_ref, Y=X, num_bins=self.num_bins, **self.kwargs)
        distance = DistanceResult(distance=js)
        return distance

    @staticmethod
    def _js(
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
        js = jensenshannon(p=X_ref_rvs, q=X_rvs, **kwargs)
        return js
