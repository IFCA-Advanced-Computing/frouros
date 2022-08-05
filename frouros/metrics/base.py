"""Metrics base module."""

import abc
from typing import Callable, Optional
import numpy as np  # type: ignore

from frouros.utils.decorators import check_func_parameters


class BaseMetric(abc.ABC):
    """Abstract class representing a metric."""

    def __init__(self, error_scorer: Callable, name: Optional[str] = None) -> None:
        """Init method.

        :param error_scorer: error scorer function
        :type error_scorer: Callable
        :param name: metric´s name
        :type name: Optional[str]
        """
        self.error_scorer = error_scorer  # type: ignore
        self.name = type(self).__name__ if name is None else name

    @property
    def error_scorer(self) -> Callable:
        """Error scorer property.

        :return: error scorer function
        :rtype: Callable
        """
        return self._error_scorer

    @error_scorer.setter  # type: ignore
    @check_func_parameters
    def error_scorer(self, value: Callable) -> None:
        """Error scorer setter.

        :param value: value to be set
        :type value: Callable
        """
        self._error_scorer = value

    @property
    def name(self) -> str:
        """Metrics' name property.

        :return: metrics' name
        :rtype: str
        """
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Metrics' name setter.

        :param value: value to be set
        :type value: str
        :raises TypeError: Type error exception
        """
        if not isinstance(value, str):
            raise TypeError("value must be of type str.")
        self._name = value

    @abc.abstractmethod
    def __call__(
        self, y_true: np.ndarray, y_pred: np.ndarray, *args, **kwargs
    ) -> float:
        """__call__ method that calculates the metric error.

        :param y_true ground truth values
        :type y_true: numpy.ndarray
        :param y_pred: predicted values
        :type y_pred: numpy.ndarray
        :return: cumulative error
        :rtype: Union[int, float]
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Abstract method that resets the metric."""
