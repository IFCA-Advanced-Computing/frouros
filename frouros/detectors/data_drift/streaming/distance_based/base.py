"""Base data drift distance based module."""

import abc
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np  # type: ignore

from frouros.detectors.data_drift.base import BaseResult
from frouros.detectors.data_drift.streaming.base import (
    BaseDataDriftStreaming,
)


class DistanceResult(BaseResult):
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


class BaseDistanceBased(BaseDataDriftStreaming):
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

    def compare(
        self,
        X: np.ndarray,  # noqa: N803
    ) -> Tuple[Optional[DistanceResult], Dict[str, Any]]:
        """Compare detector.

        :param X: data to use to compare the detector
        :type X: np.ndarray
        :return: update result
        :rtype: Tuple[Optional[DistanceResult], Dict[str, Any]]
        """
        result = self._compare(X=X)
        return result

    @abc.abstractmethod
    def _compare(
        self,
        X: np.ndarray,  # noqa: N803
    ) -> Tuple[Optional[DistanceResult], Dict[str, Any]]:
        pass
