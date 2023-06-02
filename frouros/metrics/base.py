"""Base metrics module."""

import abc
from typing import Optional


class BaseMetric(abc.ABC):
    """Abstract class representing a metric."""

    def __init__(self, name: Optional[str] = None) -> None:
        """Init method.

        :param name: name value
        :type name: Optional[str]
        """
        self.name = type(self).__name__ if name is None else name

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
        self,
        error_value: float,
    ) -> float:
        """__call__ method that updates the metric error.

        :param error_value: error value
        :type error_value: float
        :return: cumulative error
        :rtype: Union[int, float]
        """

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset method."""

    def __repr__(self) -> str:
        """Repr method.

        :return: repr value
        :rtype: str
        """
        return f"{self.__class__.__name__}(name='{self.name}')"
