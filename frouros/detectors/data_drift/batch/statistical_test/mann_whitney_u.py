"""Mann-Whitney U test module."""

from typing import Any, Optional, Union

import numpy as np
from scipy.stats import mannwhitneyu

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.base import NumericalData, UnivariateData
from frouros.detectors.data_drift.batch.statistical_test.base import (
    BaseStatisticalTest,
    StatisticalResult,
)


class MannWhitneyUTest(BaseStatisticalTest):
    """Mann–Whitney U test [mann1947test]_ detector.

    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]]

    :Note:
    - Passing additional arguments to `scipy.stats.mannwhitneyu <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html>`__ can be done using :func:`compare` kwargs.

    :References:

    .. [mann1947test] Mann, Henry B., and Donald R. Whitney.
        "On a test of whether one of two random variables is stochastically larger than
        the other."
        The annals of mathematical statistics (1947): 50-60.

    :Example:

    >>> from frouros.detectors.data_drift import MannWhitneyUTest
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> X = np.random.normal(loc=0, scale=1, size=100)
    >>> Y = np.random.normal(loc=1, scale=1, size=100)
    >>> detector = MannWhitneyUTest()
    >>> _ = detector.fit(X=X)
    >>> detector.compare(X=Y)[0]
    StatisticalResult(statistic=2139.0, p_value=2.7623373527697943e-12)
    """  # noqa: E501  # pylint: disable=line-too-long

    def __init__(  # noqa: D107
        self,
        callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]] = None,
    ) -> None:
        super().__init__(
            data_type=NumericalData(),
            statistical_type=UnivariateData(),
            callbacks=callbacks,
        )

    @staticmethod
    def _statistical_test(
        X_ref: np.ndarray,  # noqa: N803
        X: np.ndarray,
        **kwargs: Any,
    ) -> StatisticalResult:
        test = mannwhitneyu(  # pylint: disable=unexpected-keyword-arg
            x=X_ref,
            y=X,
            alternative=kwargs.get("alternative", "two-sided"),
            nan_policy=kwargs.get("nan_policy", "raise"),
            **kwargs,
        )
        test = StatisticalResult(
            statistic=test.statistic,
            p_value=test.pvalue,
        )
        return test
