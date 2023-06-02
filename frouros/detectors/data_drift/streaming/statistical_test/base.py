"""Data drift statistical test base module."""

import abc
from typing import Optional, Union

import numpy as np  # type: ignore

from frouros.detectors.data_drift.base import BaseResult
from frouros.detectors.data_drift.streaming.base import (
    BaseDataDriftStreaming,
)


class StatisticalResult(BaseResult):
    """Statistical result class."""

    def __init__(
        self,
        statistic: Union[int, float],
        p_value: Union[int, float],
    ) -> None:
        """Init method.

        :param statistic: statistic value
        :type statistic: Union[int, float]
        :param p_value: p-value
        :type p_value: Union[int, float]
        """
        super().__init__()
        self.statistic = statistic
        self.p_value = p_value

    @property
    def statistic(self) -> Union[int, float]:
        """Statistic value property.

        :return: statistic value
        :rtype: Union[int, float]
        """
        return self._statistic

    @statistic.setter
    def statistic(self, value: Union[int, float]) -> None:
        """Statistic value setter.

        :param value: value to be set
        :type value: Union[int, float]
        """
        self._statistic = value

    @property
    def p_value(self) -> Union[int, float]:
        """P-value property.

        :return: p-value
        :rtype: Union[int, float]
        """
        return self._p_value

    @p_value.setter
    def p_value(self, value: Union[int, float]) -> None:
        """P-value setter.

        :param value: value to be set
        :type value: Union[int, float]
        """
        if not 0 <= value <= 1:
            raise ValueError("p-value must be between 0 and 1.")
        self._p_value = value


class BaseStatisticalTest(BaseDataDriftStreaming):
    """Abstract class representing a statistical test."""

    @abc.abstractmethod
    def _fit(self, X: np.ndarray) -> None:  # noqa: N803
        pass

    @abc.abstractmethod
    def _reset(self) -> None:
        pass

    @abc.abstractmethod
    def _update(self, value: Union[int, float]) -> Optional[StatisticalResult]:
        pass

    @staticmethod
    @abc.abstractmethod
    def _statistical_test(
        X_ref: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> StatisticalResult:
        pass
