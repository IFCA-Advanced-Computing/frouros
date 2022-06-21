"""Unsupervised distance based base module."""

import abc
from typing import Tuple, Union

import numpy as np  # type: ignore
from scipy.stats import rv_histogram  # type: ignore

from frouros.unsupervised.base import UnivariateTest, UnsupervisedBaseEstimator


class DistanceBasedEstimator(UnsupervisedBaseEstimator):
    """Abstract class representing a distance based estimator."""

    def _apply_method(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Union[Tuple[float, float], float]:
        distance = self._distance(X_ref_=X_ref_, X=X, **kwargs)
        return distance

    @abc.abstractmethod
    def _distance(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Union[Tuple[float, float], float]:
        pass


class DistanceProbabilityBasedEstimator(DistanceBasedEstimator):
    """Abstract class representing a distance probability based estimator."""

    def __init__(self, num_points: int = 1000) -> None:
        """Init method.

        :param num_points: number of points in which to divide data
        :type num_points: int
        """
        super().__init__(test_type=UnivariateTest())
        self.num_points = num_points

    @property
    def num_points(self) -> int:
        """Number of points property.

        :return: number of points in which to divide data
        :rtype: int
        """
        return self._num_points

    @num_points.setter
    def num_points(self, value: int) -> None:
        """Number of points setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 1:
            raise ValueError("value must be greater than 0.")
        self._num_points = value

    @abc.abstractmethod
    def _distance(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Union[Tuple[float, float], float]:
        pass

    def _calculate_probabilities(
        self, X_ref_: np.ndarray, X: np.ndarray  # noqa: N803
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_ref_rv_histogram = rv_histogram(  # noqa: N806
            np.histogram(X_ref_, bins="auto")
        )
        X_rv_histogram = rv_histogram(np.histogram(X, bins="auto"))  # noqa: N806
        X_merge = np.concatenate([X_ref_, X])  # noqa: N806
        points = np.linspace(np.min(X_merge), np.max(X_merge), self.num_points)
        X_ref_rvs = [X_ref_rv_histogram.pdf(point) for point in points]  # noqa: N806
        X_rvs = [X_rv_histogram.pdf(point) for point in points]  # noqa: N806
        return X_ref_rvs, X_rvs
