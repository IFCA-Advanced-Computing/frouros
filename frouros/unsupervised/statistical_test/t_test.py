"""T-test module."""

from typing import Tuple

import numpy as np  # type: ignore
from scipy.stats import ttest_ind  # type: ignore

from frouros.unsupervised.base import NumericalData, UnivariateTestType
from frouros.unsupervised.statistical_test.base import StatisticalTestBaseEstimator


class TTest(StatisticalTestBaseEstimator):
    """T-test algorithm class."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__(data_type=NumericalData(), test_type=UnivariateTestType())

    def _statistical_test(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Tuple[float, float]:
        test = ttest_ind(
            a=X_ref_, b=X, equal_var=False, alternative="two-sided", **kwargs
        )
        return test.statistic, test.pvalue
