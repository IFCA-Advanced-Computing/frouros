"""PSI (Population Stability Index) module."""

import sys
from typing import Any, Optional, Union

import numpy as np

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.base import UnivariateData
from frouros.detectors.data_drift.batch.distance_based.base import (
    BaseDistanceBasedBins,
    DistanceResult,
)


class PSI(BaseDistanceBasedBins):
    """PSI (Population Stability Index) [wu2010enterprise]_ detector.

    :param num_bins: number of bins in which to divide probabilities, defaults to 10
    :type num_bins: int
    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]]

    :References:

    .. [wu2010enterprise] Wu, Desheng, and David L. Olson.
        "Enterprise risk management: coping with model risk in a large bank."
        Journal of the Operational Research Society 61.2 (2010): 179-190.

    :Example:

    >>> from frouros.detectors.data_drift import PSI
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> X = np.random.normal(loc=0, scale=1, size=100)
    >>> Y = np.random.normal(loc=1, scale=1, size=100)
    >>> detector = PSI(num_bins=20)
    >>> _ = detector.fit(X=X)
    >>> detector.compare(X=Y)[0]
    DistanceResult(distance=134.95409065116183)
    """

    def __init__(  # noqa: D107
        self,
        num_bins: int = 10,
        callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]] = None,
    ) -> None:
        super().__init__(
            statistical_type=UnivariateData(),
            statistical_method=self._psi,
            statistical_kwargs={
                "num_bins": num_bins,
            },
            callbacks=callbacks,
        )
        self.num_bins = num_bins

    def _apply_method(
        self,
        X_ref: np.ndarray,  # noqa: N803
        X: np.ndarray,
        **kwargs: Any,
    ) -> DistanceResult:
        distance = self._distance_measure(X_ref=X_ref, X=X, **kwargs)
        return distance

    def _distance_measure_bins(
        self,
        X_ref: np.ndarray,  # noqa: N803
        X: np.ndarray,  # noqa: N803
    ) -> float:
        psi = self._psi(X=X_ref, Y=X, num_bins=self.num_bins)
        return psi

    @staticmethod
    def _psi(
        X: np.ndarray,  # noqa: N803
        Y: np.ndarray,  # noqa: N803
        num_bins: int,
    ) -> float:
        (  # noqa: N806
            X_percents,
            Y_percents,
        ) = BaseDistanceBasedBins._calculate_bins_values(
            X_ref=X, X=Y, num_bins=num_bins
        )
        # Replace 0.0 values with the smallest number possible
        # in order to avoid division by zero
        X_percents[X_percents == 0.0] = sys.float_info.min
        Y_percents[Y_percents == 0.0] = sys.float_info.min
        psi = np.sum((Y_percents - X_percents) * np.log(Y_percents / X_percents))
        return psi
