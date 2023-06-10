"""Mann-Whitney U test module."""

from typing import Optional, List, Union

import numpy as np  # type: ignore
from scipy.stats import mannwhitneyu  # type: ignore

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.base import NumericalData, UnivariateData
from frouros.detectors.data_drift.batch.statistical_test.base import (
    BaseStatisticalTest,
    StatisticalResult,
)


class MannWhitneyUTest(BaseStatisticalTest):
    """Mann–Whitney U test [mann1947test]_ detector.

    :References:

    .. [mann1947test] Mann, Henry B., and Donald R. Whitney.
        "On a test of whether one of two random variables is stochastically larger than
        the other."
        The annals of mathematical statistics (1947): 50-60.
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
        self, X_ref: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> StatisticalResult:
        test = mannwhitneyu(  # pylint: disable=unexpected-keyword-arg
            x=X_ref,
            y=X,
            alternative="two-sided",
            nan_policy="raise",
            **kwargs,
        )
        test = StatisticalResult(
            statistic=test.statistic,
            p_value=test.pvalue,
        )
        return test
