"""Base data drift statistical test module."""

import abc
from collections import namedtuple
from typing import Any, Tuple

import numpy as np

from frouros.detectors.data_drift.batch.base import BaseDataDriftBatch

StatisticalResult = namedtuple("StatisticalResult", ["statistic", "p_value"])


class BaseStatisticalTest(BaseDataDriftBatch):
    """Abstract class representing a statistical test."""

    def _apply_method(
        self,
        X_ref: np.ndarray,  # noqa: N803
        X: np.ndarray,
        **kwargs: Any,
    ) -> Tuple[float, float]:
        statistical_test = self._statistical_test(
            X_ref=X_ref,
            X=X,
            **kwargs,
        )
        return statistical_test

    def _compare(
        self,
        X: np.ndarray,  # noqa: N803
        **kwargs: Any,
    ) -> StatisticalResult:
        self._common_checks()  # noqa: N806
        self._specific_checks(X=X)  # noqa: N806
        result = self._get_result(X=X, **kwargs)
        return result  # type: ignore

    @staticmethod
    @abc.abstractmethod
    def _statistical_test(
        X_ref: np.ndarray,  # noqa: N803
        X: np.ndarray,
        **kwargs: Any,
    ) -> StatisticalResult:
        pass
