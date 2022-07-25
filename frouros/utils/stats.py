"""Stats module."""

import abc
from typing import Union


class IncrementalStat(abc.ABC):
    """Abstract class representing an incremental statistic."""

    @abc.abstractmethod
    def update(self, value: Union[int, float]) -> None:
        """Update abstract method."""


class Mean(IncrementalStat):
    """Incremental mean class."""

    def __init__(self) -> None:
        """Init method."""
        self.mean = 0.0
        self.num_values = 0

    @property
    def mean(self) -> float:
        """Mean property.

        :return: mean value
        :rtype: float
        """
        return self._mean

    @mean.setter
    def mean(self, value: float) -> None:
        """Mean setter.

        :param value: value to be set
        :type value: float
        """
        self._mean = value

    @property
    def num_values(self) -> int:
        """Number of values property.

        :return: number of values
        :rtype: int
        """
        return self._num_values

    @num_values.setter
    def num_values(self, value: int) -> None:
        """Number of values setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("num_values must be greater of equal than 0.")
        self._num_values = value

    def update(self, value: Union[int, float]) -> None:
        """Update the mean value sequentially.

        :param value: value to use to update the mean
        :type value: int
        :raises TypeError: Type error exception
        """
        if not isinstance(value, (int, float)):
            raise TypeError("value must be of type int or float.")
        self.num_values += 1
        self.mean += (value - self.mean) / self.num_values


class EWMA(IncrementalStat):
    """EWMA (Exponential Weighted Moving Average) class."""

    def __init__(self, alpha: float) -> None:
        """Init method.

        :param alpha:
        :type alpha: float
        """
        self.alpha = alpha
        self.one_minus_alpha = 1.0 - self.alpha
        self.mean = 0

    @property
    def alpha(self) -> float:
        """Alpha property.

        :return: alpha value
        :rtype: float
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Alpha setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError("alpha must be in the range [0, 1].")
        self._alpha = value

    @property
    def mean(self) -> float:
        """Mean property.

        :return: mean value
        :rtype: float
        """
        return self._mean

    @mean.setter
    def mean(self, value: float) -> None:
        """Mean setter.

        :param value: value to be set
        :type value: float
        """
        self._mean = value

    def update(self, value: Union[int, float]) -> None:
        """Update the mean value sequentially.

        :param value: value to use to update the mean
        :type value: int
        :raises TypeError: Type error exception
        """
        if not isinstance(value, (int, float)):
            raise TypeError("value must be of type int or float.")
        self.mean = self.alpha * value + self.one_minus_alpha * self.mean
