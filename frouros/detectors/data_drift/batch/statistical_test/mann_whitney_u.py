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

    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]]
    :param kwargs: additional keyword arguments to pass to scipy.stats.mannwhitneyu
    :type kwargs: Dict[str, Any]

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
    """

    def __init__(
        self,
        callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            data_type=NumericalData(),
            statistical_type=UnivariateData(),
            callbacks=callbacks,
        )
        self.kwargs = kwargs

    def _statistical_test(
        self, X_ref: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> StatisticalResult:
        test = mannwhitneyu(  # pylint: disable=unexpected-keyword-arg
            x=X_ref,
            y=X,
            alternative="two-sided",
            nan_policy="raise",
            **self.kwargs,
        )
        test = StatisticalResult(
            statistic=test.statistic,
            p_value=test.pvalue,
        )
        return test
