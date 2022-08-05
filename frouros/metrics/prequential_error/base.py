"""Prequential error metrics base module."""

import abc
from typing import Callable, Optional, Union
import numpy as np  # type: ignore

from frouros.metrics.base import BaseMetric


class PrequentialErrorBase(BaseMetric):
    """Abstract class representing a prequential error metric."""

    def __init__(self, error_scorer: Callable, name: Optional[str] = None) -> None:
        """Init method.

        :param error_scorer: error scorer function
        :type error_scorer: Callable
        :param name: metric´s name
        :type name: Optional[str]
        """
        super().__init__(error_scorer=error_scorer, name=name)
        self.cumulative_error = 0.0

    @property
    def cumulative_error(self) -> Union[int, float]:
        """Cumulative error property.

        :return: cumulative error value
        :rtype: Union[int, float]
        """
        return self._cumulative_error

    @cumulative_error.setter
    def cumulative_error(self, value: Union[int, float]) -> None:
        """Cumulative error setter.

        :param value: value to be set
        :type value: Union[int, float]
        """
        if not isinstance(value, (int, float)):
            raise TypeError("value must be of type int or float.")
        self._cumulative_error = value

    def reset(self) -> None:
        """Reset cumulative_error."""

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
