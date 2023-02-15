"""Data drift statistical test base module."""

import abc
from typing import Optional, Union

import numpy as np  # type: ignore

from frouros.detectors.data_drift.streaming.base import (
    DataDriftStreamingBase,
    StatisticalResult,
)


class StatisticalTestBase(DataDriftStreamingBase):
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
