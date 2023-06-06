"""JS (Jensen-Shannon distance) module."""

from typing import Any, Dict, List, Optional, Union

import numpy as np  # type: ignore
from scipy.spatial.distance import jensenshannon  # type: ignore

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.batch.distance_based.base import (
    BaseDistanceBasedProbability,
    DistanceResult,
)


class JS(BaseDistanceBasedProbability):
    """JS (Jensen-Shannon distance) [lin1991divergence]_ detector.

    :References:

    .. [lin1991divergence] Lin, Jianhua.
        "Divergence measures based on the Shannon entropy."
        IEEE Transactions on Information theory 37.1 (1991): 145-151.
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
        :param callbacks: callbacks
        :type callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]]
        """
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
        **kwargs,
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
        js = jensenshannon(p=X_ref_rvs, q=X_rvs, **kwargs)
        return js
