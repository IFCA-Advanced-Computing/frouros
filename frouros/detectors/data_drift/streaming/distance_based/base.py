"""Data drift distance based base module."""

import abc
from typing import Optional, Union

import numpy as np  # type: ignore

from frouros.detectors.data_drift.base import ResultBase
from frouros.detectors.data_drift.streaming.base import (
    DataDriftStreamingBase,
)


class DistanceResult(ResultBase):
    """Distance result class."""

    def __init__(
        self,
        distance: Union[int, float],
    ) -> None:
        """Init method.

        :param distance: distance value
        :type distance: Union[int, float]
        """
        super().__init__()
        self.distance = distance

    @property
    def distance(self) -> Union[int, float]:
        """Distance value property.

        :return: distance value
        :rtype: Union[int, float]
        """
        return self._distance

    @distance.setter
    def distance(self, value: Union[int, float]) -> None:
        """Distance value setter.

        :param value: value to be set
        :type value: Union[int, float]
        """
        self._distance = value


class DistanceBasedBase(DataDriftStreamingBase):
    """Abstract class representing a distance based."""

    @abc.abstractmethod
    def _fit(self, X: np.ndarray) -> None:  # noqa: N803
        pass

    @abc.abstractmethod
    def _reset(self) -> None:
        pass

    @abc.abstractmethod
    def _update(self, value: Union[int, float]) -> Optional[DistanceResult]:
        pass
