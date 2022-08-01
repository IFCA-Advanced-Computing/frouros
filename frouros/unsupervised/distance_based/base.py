"""Unsupervised distance based base module."""

import abc
from collections import namedtuple
from typing import Optional, List, Tuple, Union

import numpy as np  # type: ignore
from scipy.stats import rv_histogram  # type: ignore

from frouros.unsupervised.base import (
    BaseDataType,
    BaseStatisticalType,
    NumericalData,
    UnivariateType,
    UnsupervisedBaseEstimator,
)

DistanceResult = namedtuple("DistanceResult", ["distance"])
DistanceTestResult = namedtuple("DistanceTestResult", ["distance", "p_value"])


class DistanceBasedEstimator(UnsupervisedBaseEstimator):
    """Abstract class representing a distance based estimator."""

    def __init__(
        self, data_type: BaseDataType, statistical_type: BaseStatisticalType
    ) -> None:
        """Init method.

        :param data_type: data type
        :type data_type: BaseDataType
        :param statistical_type: statistical type
        :type statistical_type: BaseStatisticalType
        """
        super().__init__(data_type=data_type, statistical_type=statistical_type)
        self.X_ref_ = None  # type: ignore
        self.distance: Optional[
            Union[List[DistanceResult], List[DistanceTestResult]]
        ] = None

    @property
    def distance(
        self,
    ) -> Optional[Union[List[DistanceResult], List[DistanceTestResult]]]:
        """Distance results property.

        :return: distance results
        :rtype: Optional[Union[List[DistanceResult], List[DistanceTestResult]]]
        """
        return self._distance

    @distance.setter
    def distance(
        self, value: Optional[Union[List[DistanceResult], List[DistanceTestResult]]]
    ) -> None:
        """Distance results setter.

        :param value: value to be set
        :type value: Optional[Union[List[DistanceResult], List[DistanceTestResult]]]
        """
        self._distance = value

    def _apply_method(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Union[DistanceResult, DistanceTestResult]:
        distance = self._distance_measure(X_ref_=X_ref_, X=X, **kwargs)
        return distance

    def transform(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray = None,  # pylint: disable=W0613
        **kwargs,
    ) -> np.ndarray:
        """Transform values.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :type y: numpy.ndarray
        :return: transformed feature data
        :rtype: numpy.ndarray
        """
        X = self._common_checks(X=X)  # noqa: N806
        self._specific_checks(X=X)  # noqa: N806
        self.distance = self._get_result(X=X, **kwargs)  # type: ignore
        return X

    @abc.abstractmethod
    def _distance_measure(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Union[DistanceResult, DistanceTestResult]:
        pass


class DistanceProbabilityBasedEstimator(DistanceBasedEstimator):
    """Abstract class representing a distance probability based estimator."""

    def __init__(self, num_bins: int = 100) -> None:
        """Init method.

        :param num_bins: number of bins in which to divide probabilities
        :type num_bins: int
        """
        super().__init__(data_type=NumericalData(), statistical_type=UnivariateType())
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
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> DistanceResult:
        pass

    def _calculate_probabilities(
        self, X_ref_: np.ndarray, X: np.ndarray  # noqa: N803
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_ref_rv_histogram = rv_histogram(  # noqa: N806
            np.histogram(X_ref_, bins="auto")
        )
        X_rv_histogram = rv_histogram(np.histogram(X, bins="auto"))  # noqa: N806
        X_merge = np.concatenate([X_ref_, X])  # noqa: N806
        bins = np.linspace(np.min(X_merge), np.max(X_merge), self.num_bins)
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
