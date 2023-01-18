"""Data drift statistical test base module."""

import abc
from collections import namedtuple
from typing import Optional, List, Tuple

import numpy as np  # type: ignore

from frouros.data_drift.batch.base import (
    DataTypeBase,
    DataDriftBatchBase,
    StatisticalTypeBase,
)


TestResult = namedtuple("TestResult", ["statistic", "p_value"])


class StatisticalTestBase(DataDriftBatchBase):
    """Abstract class representing a statistical test."""

    def __init__(
        self, data_type: DataTypeBase, statistical_type: StatisticalTypeBase
    ) -> None:
        """Init method.

        :param data_type: data type
        :type data_type: BaseDataType
        """
        super().__init__(data_type=data_type, statistical_type=statistical_type)
        self.test: Optional[List[TestResult]] = None

    @property
    def test(self) -> Optional[List[TestResult]]:
        """Test results property.

        :return: test results
        :rtype: Optional[List[TestResult]]
        """
        return self._test

    @test.setter
    def test(self, value: Optional[List[TestResult]]) -> None:
        """Test results setter.

        :param value: value to be set
        :type value: Optional[List[TestResult]]
        """
        self._test = value

    def _apply_method(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Tuple[float, float]:
        statistical_test = self._statistical_test(X_ref_=X_ref_, X=X, **kwargs)
        return statistical_test

    def compare(
        self,
        X: np.ndarray,  # noqa: N803
        **kwargs,
    ) -> None:
        """Transform values.

        :param X: feature data
        :type X: numpy.ndarray
        """
        self._common_checks(X=X)  # noqa: N806
        self._specific_checks(X=X)  # noqa: N806
        result = self._get_result(X=X, **kwargs)  # type: ignore
        return result

    @abc.abstractmethod
    def _statistical_test(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Tuple[float, float]:
        pass
