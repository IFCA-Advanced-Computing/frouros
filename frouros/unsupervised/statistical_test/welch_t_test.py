"""Welch's T-test module."""

import numpy as np  # type: ignore
from scipy.stats import ttest_ind  # type: ignore

from frouros.unsupervised.base import NumericalData, UnivariateType
from frouros.unsupervised.statistical_test.base import (
    StatisticalTestBaseEstimator,
    TestResult,
)


class WelchTTest(StatisticalTestBaseEstimator):
    """Welch's T-test algorithm class."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__(data_type=NumericalData(), statistical_type=UnivariateType())

    def _statistical_test(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> TestResult:
        test = ttest_ind(
            a=X_ref_, b=X, equal_var=False, alternative="two-sided", **kwargs
        )
        test = TestResult(statistic=test.statistic, p_value=test.pvalue)
        return test
