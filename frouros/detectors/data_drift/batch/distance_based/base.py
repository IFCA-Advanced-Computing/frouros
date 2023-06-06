"""Base data drift batch distance based module."""

import abc
from collections import namedtuple
from typing import Any, Callable, Dict, Optional, List, Tuple, Union

import numpy as np  # type: ignore
from scipy.stats import rv_histogram  # type: ignore

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.base import (
    NumericalData,
    BaseStatisticalType,
    UnivariateData,
)
from frouros.detectors.data_drift.batch.base import BaseDataDriftBatch

DistanceResult = namedtuple("DistanceResult", ["distance"])


class BaseDistanceBased(BaseDataDriftBatch):
    """Abstract class representing a distance based detector."""

    def __init__(
        self,
        statistical_type: BaseStatisticalType,
        statistical_method: Callable,
        statistical_kwargs: Dict[str, Any],
        callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]] = None,
    ) -> None:
        """Init method.

        :param statistical_type: statistical type
        :type statistical_type: BaseStatisticalType
        :param statistical_method: statistical method
        :type statistical_method: Callable
        :param statistical_kwargs: statistical kwargs
        :type statistical_kwargs: Dict[str, Any]
        :param callbacks: callbacks
        :type callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]]
        """
        super().__init__(
            data_type=NumericalData(),
            statistical_type=statistical_type,
            callbacks=callbacks,
        )
        self.statistical_method = statistical_method
        self.statistical_kwargs = statistical_kwargs

    @property
    def statistical_method(self) -> Callable:
        """Statistical method property.

        :return: statistical method
        :rtype: Callable
        """
        return self._statistical_method

    @statistical_method.setter
    def statistical_method(self, value: Callable) -> None:
        """Statistical method setter.

        :param value: value to be set
        :type value: Callable
        :raises TypeError: Type error exception
        """
        if not isinstance(value, Callable):  # type: ignore
            raise TypeError("value must be of type Callable.")
        self._statistical_method = value

    @property
    def statistical_kwargs(self) -> Dict[str, Any]:
        """Statistical kwargs property.

        :return: statistical kwargs
        :rtype: Dict[str, Any]
        """
        return self._statistical_kwargs

    @statistical_kwargs.setter
    def statistical_kwargs(self, value: Dict[str, Any]) -> None:
        """Statistical kwargs setter.

        :param value: value to be set
        :type value: Dict[str, Any]
        """
        self._statistical_kwargs = value

    def _apply_method(
        self, X_ref: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> DistanceResult:
        distance = self._distance_measure(X_ref=X_ref, X=X, **kwargs)
        return distance

    def _compare(
        self,
        X: np.ndarray,  # noqa: N803
        **kwargs,
    ) -> DistanceResult:
        self._common_checks()  # noqa: N806
        self._specific_checks(X=X)  # noqa: N806
        distance = self._get_result(X=X, **kwargs)  # type: ignore
        return distance  # type: ignore

    @abc.abstractmethod
    def _distance_measure(
        self,
        X_ref: np.ndarray,  # noqa: N803
        X: np.ndarray,  # noqa: N803
        **kwargs,
    ) -> DistanceResult:
        pass


class BaseDistanceBasedBins(BaseDistanceBased):
    """Abstract class representing a distance based bins detector."""

    def __init__(
        self,
        statistical_method,
        statistical_kwargs,
        callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]] = None,
        num_bins: int = 10,
    ) -> None:
        """Init method.

        :param statistical_method: statistical method
        :type statistical_method: Callable
        :param statistical_kwargs: statistical kwargs
        :type statistical_kwargs: Dict[str, Any]
        :param callbacks: callbacks
        :type callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]]
        :param num_bins: number of bins in which to divide probabilities
        :type num_bins: int
        """
        super().__init__(
            statistical_type=UnivariateData(),
            statistical_method=statistical_method,
            statistical_kwargs={**statistical_kwargs, "num_bins": num_bins},
            callbacks=callbacks,
        )
        self.num_bins = num_bins

    @property
    def num_bins(self) -> int:
        """Number of bins property.

        :return: number of bins in which to divide probabilities
        :rtype: int
        """
        return self._num_bins

    @num_bins.setter
    def num_bins(self, value: int) -> None:
        """Number of bins setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 1:
            raise ValueError("value must be greater than 0.")
        self._num_bins = value

    def _distance_measure(
        self,
        X_ref: np.ndarray,  # noqa: N803
        X: np.ndarray,  # noqa: N803
        **kwargs,
    ) -> DistanceResult:
        distance_bins = self._distance_measure_bins(X_ref=X_ref, X=X)
        distance = DistanceResult(distance=distance_bins)
        return distance

    @staticmethod
    def _calculate_bins_values(
        X_ref: np.ndarray, X: np.ndarray, num_bins: int = 10  # noqa: N803
    ) -> np.ndarray:
        bins = np.histogram(np.hstack((X_ref, X)), bins=num_bins)[  # get the bin edges
            1
        ]
        X_ref_percents = (  # noqa: N806
            np.histogram(a=X_ref, bins=bins)[0] / X_ref.shape[0]
        )  # noqa: N806
        X_percents = np.histogram(a=X, bins=bins)[0] / X.shape[0]  # noqa: N806
        return X_ref_percents, X_percents

    @abc.abstractmethod
    def _distance_measure_bins(
        self,
        X_ref: np.ndarray,  # noqa: N803
        X: np.ndarray,  # noqa: N803
    ) -> float:
        pass


class BaseDistanceBasedProbability(BaseDistanceBased):
    """Abstract class representing a distance based probability detector."""

    def __init__(
        self,
        statistical_method,
        statistical_kwargs,
        callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]] = None,
        num_bins: int = 10,
    ) -> None:
        """Init method.

        :param statistical_method: statistical method
        :type statistical_method: Callable
        :param statistical_kwargs: statistical kwargs
        :type statistical_kwargs: Dict[str, Any]
        :param callbacks: callbacks
        :type callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]]
        :param num_bins: number of bins in which to divide probabilities
        :type num_bins: int
        """
        super().__init__(
            statistical_type=UnivariateData(),
            statistical_method=statistical_method,
            statistical_kwargs=statistical_kwargs,
            callbacks=callbacks,
        )
        self.num_bins = num_bins

    @property
    def num_bins(self) -> int:
        """Number of bins property.

        :return: number of bins in which to divide probabilities
        :rtype: int
        """
        return self._num_bins

    @num_bins.setter
    def num_bins(self, value: int) -> None:
        """Number of bins setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 1:
            raise ValueError("value must be greater than 0.")
        self._num_bins = value

    @abc.abstractmethod
    def _distance_measure(
        self,
        X_ref: np.ndarray,  # noqa: N803
        X: np.ndarray,  # noqa: N803
        **kwargs,
    ) -> DistanceResult:
        pass

    @staticmethod
    def _calculate_probabilities(
        X_ref: np.ndarray,  # noqa: N803
        X: np.ndarray,
        num_bins: int,  # noqa: N803
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_ref_rv_histogram = rv_histogram(  # noqa: N806
            np.histogram(X_ref, bins="auto")
        )
        X_rv_histogram = rv_histogram(np.histogram(X, bins="auto"))  # noqa: N806
        X_merge = np.concatenate([X_ref, X])  # noqa: N806
        bins = np.linspace(np.min(X_merge), np.max(X_merge), num_bins)
        X_ref_rvs = [  # noqa: N806
            X_ref_rv_histogram.cdf(bins[i])
            - X_ref_rv_histogram.cdf(bins[i - 1])  # noqa: N806
            for i in range(1, len(bins[1:]) + 1)
        ]
        X_rvs = [  # noqa: N806
            X_rv_histogram.cdf(bins[i]) - X_rv_histogram.cdf(bins[i - 1])  # noqa: N806
            for i in range(1, len(bins[1:]) + 1)
        ]
        return X_ref_rvs, X_rvs
