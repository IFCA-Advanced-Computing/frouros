"""Base data drift statistical test module."""

import abc
from collections import namedtuple
from typing import Tuple

import numpy as np  # type: ignore

from frouros.detectors.data_drift.batch.base import BaseDataDriftBatch

StatisticalResult = namedtuple("StatisticalResult", ["statistic", "p_value"])


class BaseStatisticalTest(BaseDataDriftBatch):
    """Abstract class representing a statistical test."""

    def _apply_method(
        self, X_ref: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Tuple[float, float]:
        statistical_test = self._statistical_test(X_ref=X_ref, X=X, **kwargs)
        return statistical_test

    def _compare(
        self,
        X: np.ndarray,  # noqa: N803
        **kwargs,
    ) -> StatisticalResult:
        self._common_checks()  # noqa: N806
        self._specific_checks(X=X)  # noqa: N806
        result = self._get_result(X=X, **kwargs)
        return result  # type: ignore

    @abc.abstractmethod
    def _statistical_test(
        self, X_ref: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Tuple[float, float]:
        pass
