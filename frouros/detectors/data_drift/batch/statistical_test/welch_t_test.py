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

    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]]
    :param kwargs: additional keyword arguments to pass to scipy.stats.ttest_ind
    :type kwargs: Dict[str, Any]

    :References:

    .. [welch1947generalization] Welch, Bernard L.
        "The generalization of ‘STUDENT'S’problem when several different population
        varlances are involved."
        Biometrika 34.1-2 (1947): 28-35.

    :Example:

    >>> from frouros.detectors.data_drift import WelchTTest
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> X = np.random.normal(loc=0, scale=1, size=100)
    >>> Y = np.random.normal(loc=1, scale=1, size=100)
    >>> detector = WelchTTest()
    >>> _ = detector.fit(X=X)
    >>> detector.compare(X=Y)[0]
    StatisticalResult(statistic=-7.651304662806378, p_value=8.685225410826823e-13)
    """

    def __init__(  # noqa: D107
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
            **self.kwargs,
        )
        test = StatisticalResult(
            statistic=test.statistic,
            p_value=test.pvalue,
        )
        return test
