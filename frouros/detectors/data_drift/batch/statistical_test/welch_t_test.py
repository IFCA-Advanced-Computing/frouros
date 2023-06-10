"""Welch's t-test module."""

from typing import Optional, List, Union

import numpy as np  # type: ignore
from scipy.stats import ttest_ind  # type: ignore

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.base import NumericalData, UnivariateData
from frouros.detectors.data_drift.batch.statistical_test.base import (
    BaseStatisticalTest,
    StatisticalResult,
)


class WelchTTest(BaseStatisticalTest):
    """Welch's t-test [welch1947generalization]_ detector.

    :References:

    .. [welch1947generalization] Welch, Bernard L.
        "The generalization of ‘STUDENT'S’problem when several different population
        varlances are involved."
        Biometrika 34.1-2 (1947): 28-35.
    """

    def __init__(
        self,
        callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]] = None,
    ) -> None:
        """Init method.

        :param callbacks: callbacks
        :type callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]]
        """
        super().__init__(
            data_type=NumericalData(),
            statistical_type=UnivariateData(),
            callbacks=callbacks,
        )

    def _statistical_test(
        self,
        X_ref: np.ndarray,  # noqa: N803
        X: np.ndarray,
        **kwargs,
    ) -> StatisticalResult:
        test = ttest_ind(
            a=X_ref,
            b=X,
            equal_var=False,
            alternative="two-sided",
            **kwargs,
        )
        test = StatisticalResult(
            statistic=test.statistic,
            p_value=test.pvalue,
        )
        return test
