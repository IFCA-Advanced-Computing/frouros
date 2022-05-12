"""Supervised statistical test base module."""

import abc
from inspect import signature
from typing import (  # noqa: TYP001
    Callable,
    Dict,
    Optional,
    Union,
)

from sklearn.base import BaseEstimator  # type: ignore
import numpy as np  # type: ignore

from frouros.supervised.base import TargetDelayEstimator, SupervisedBaseConfig
from frouros.supervised.exceptions import ArgumentError


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
    def error_scorer(self) -> Callable:
        """Error scorer property.

        :return: error scorer function
        :rtype: Callable
        """
        return self._error_scorer

    @error_scorer.setter
    def error_scorer(self, value: Callable) -> None:
        """Error scorer setter.

        :param value: value to be set
        :type value: Callable
        :raises ArgumentError: Argument error exception
        """
        func_parameters = signature(value).parameters
        if "y_true" not in func_parameters or "y_pred" not in func_parameters:
            raise ArgumentError("value function must have y_true and y_pred arguments.")
        self._error_scorer = value

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
