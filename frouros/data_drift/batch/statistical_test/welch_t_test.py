"""Welch's T-test module."""

from typing import Optional, List, Union

import numpy as np  # type: ignore
from scipy.stats import ttest_ind  # type: ignore

from frouros.callbacks import Callback
from frouros.data_drift.base import NumericalData, UnivariateData
from frouros.data_drift.batch.statistical_test.base import (
    StatisticalTestBase,
    StatisticalResult,
)


class WelchTTest(StatisticalTestBase):
    """Welch's T-test algorithm class."""

    def __init__(
        self, callbacks: Optional[Union[Callback, List[Callback]]] = None
    ) -> None:
        """Init method.

        :param callbacks: callbacks
        :type callbacks: Optional[Union[Callback, List[Callback]]]
        """
        super().__init__(
            data_type=NumericalData(),
            statistical_type=UnivariateData(),
            callbacks=callbacks,
        )

    def _statistical_test(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> StatisticalResult:
        test = ttest_ind(
            a=X_ref_, b=X, equal_var=False, alternative="two-sided", **kwargs
        )
        test = StatisticalResult(statistic=test.statistic, p_value=test.pvalue)
        return test
