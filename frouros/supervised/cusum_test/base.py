"""Supervised statistical test base module."""

import abc
from typing import (  # noqa: TYP001
    Callable,
    Dict,
    Optional,
    Union,
)

from sklearn.base import BaseEstimator  # type: ignore
import numpy as np  # type: ignore

from frouros.supervised.base import TargetDelayEstimator, SupervisedBaseConfig


class CUSUMTestConfig(SupervisedBaseConfig):
    """Class representing a CUSUM (cumulative sum) test configuration class ."""


class CUSUMTestEstimator(TargetDelayEstimator):
    """Page Hinkley test algorithm class."""

    def __init__(
        self,
        estimator: BaseEstimator,
        error_scorer: Callable,
        config: CUSUMTestConfig,
    ) -> None:
        """Init method.

        :param estimator: sklearn estimator
        :type estimator: BaseEstimator
        :param error_scorer: error scorer function
        :type error_scorer: Callable
        :param config: configuration parameters
        :type config: CUSUMTestConfig
        """
        super().__init__(estimator=estimator, config=config)
        self.error_scorer = error_scorer
        self.mean_error_rate = 0.0
        self.sum_ = 0.0

    @property
    def mean_error_rate(self) -> float:
        """Mean error rate property.

        :return: mean error rate to use
        :rtype: float
        """
        return self._mean_error_rate

    @mean_error_rate.setter
    def mean_error_rate(self, value: float) -> None:
        """Mean error rate setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("mean_error_rate must be great or equal than 0.")
        self._mean_error_rate = value

    @property
    def sum_(self) -> float:
        """Sum count property.

        :return: sum count value
        :rtype: float
        """
        return self._sum

    @sum_.setter
    def sum_(self, value: float) -> None:
        """Sum count setter.

        :param value: value to be set
        :type value: float
        """
        self._sum = value

    @abc.abstractmethod
    def update(self, y: np.array) -> Dict[str, Optional[Union[float, bool]]]:
        """Update drift detector.

        :param y: input data
        :type y: numpy.ndarray
        :return predicted values
        :rtype: Dict[str, Optional[Union[float, bool]]]
        """
