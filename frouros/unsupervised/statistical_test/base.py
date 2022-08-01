"""Unsupervised statistical test base module."""

import abc
from collections import namedtuple
from typing import Optional, List, Tuple

import numpy as np  # type: ignore

from frouros.unsupervised.base import (
    BaseDataType,
    BaseStatisticalType,
    UnsupervisedBaseEstimator,
)


TestResult = namedtuple("TestResult", ["statistic", "p_value"])


class StatisticalTestBaseEstimator(UnsupervisedBaseEstimator):
    """Abstract class representing a statistical test estimator."""

    def __init__(
        self, data_type: BaseDataType, statistical_type: BaseStatisticalType
    ) -> None:
        """Init method.

        :param data_type: data type
        :type data_type: BaseDataType
        :param statistical_type: statistical type
        :type statistical_type: BaseStatisticalType
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

    def transform(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray = None,  # pylint: disable=W0613
        **kwargs,
    ) -> np.ndarray:
        """Transform values.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :type y: numpy.ndarray
        :return: transformed feature data
        :rtype: numpy.ndarray
        """
        X = self._common_checks(X=X)  # noqa: N806
        self._specific_checks(X=X)  # noqa: N806
        self.test = self._get_result(X=X, **kwargs)  # type: ignore
        return X

    @abc.abstractmethod
    def _statistical_test(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Tuple[float, float]:
        pass
