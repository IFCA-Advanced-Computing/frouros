"""Anderson-Darling test module."""

from typing import Any, Optional, Union

import numpy as np
from scipy.stats import anderson_ksamp

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.base import NumericalData, UnivariateData
from frouros.detectors.data_drift.batch.statistical_test.base import (
    BaseStatisticalTest,
    StatisticalResult,
)


class AndersonDarlingTest(BaseStatisticalTest):
    """Anderson-Darling test [scholz1987k]_ detector.

    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]]

    :Note:
    - Passing additional arguments to `scipy.stats.anderson_ksamp <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson_ksamp.html>`__ can be done using :func:`compare` kwargs.
    - p-values are bounded between 0.001 and 0.25 according to `scipy.stats.anderson_ksamp <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson_ksamp.html>`__.

    :References:

    .. [scholz1987k] Scholz, Fritz W., and Michael A. Stephens.
        "K-sample Anderson–Darling tests."
        Journal of the American Statistical Association 82.399 (1987): 918-924.

    :Example:

    >>> from frouros.detectors.data_drift import AndersonDarlingTest
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> X = np.random.normal(loc=0, scale=1, size=100)
    >>> Y = np.random.normal(loc=1, scale=1, size=100)
    >>> detector = AndersonDarlingTest()
    >>> _ = detector.fit(X=X)
    >>> detector.compare(X=Y)[0]
    StatisticalResult(statistic=32.40316586267425, p_value=0.001)
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
        test = anderson_ksamp(
            samples=[
                X_ref,
                X,
            ],
            **kwargs,
        )
        test = StatisticalResult(
            statistic=test.statistic,
            p_value=test.pvalue,
        )
        return test
