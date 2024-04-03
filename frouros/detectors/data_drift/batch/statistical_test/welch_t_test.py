"""Welch's t-test module."""

from typing import Any, Optional, Union

import numpy as np
from scipy.stats import ttest_ind

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.base import NumericalData, UnivariateData
from frouros.detectors.data_drift.batch.statistical_test.base import (
    BaseStatisticalTest,
    StatisticalResult,
)


class WelchTTest(BaseStatisticalTest):
    """Welch's t-test [welch1947generalization]_ detector.

    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]]

    :Note:
    - Passing additional arguments to `scipy.stats.ttest_ind <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html>`__ can be done using :func:`compare` kwargs.

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
        test = ttest_ind(
            a=X_ref,
            b=X,
            equal_var=False,
            alternative=kwargs.get("alternative", "two-sided"),
            **kwargs,
        )
        test = StatisticalResult(
            statistic=test.statistic,
            p_value=test.pvalue,
        )
        return test
