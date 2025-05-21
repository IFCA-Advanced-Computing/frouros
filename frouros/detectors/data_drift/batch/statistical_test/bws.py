"""BWSTest (Baumgartner-Weiss-Schindler test) module."""

from typing import Any, Optional, Union

import numpy as np
from scipy.stats import bws_test

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.base import NumericalData, UnivariateData
from frouros.detectors.data_drift.batch.statistical_test.base import (
    BaseStatisticalTest,
    StatisticalResult,
)


class BWSTest(BaseStatisticalTest):
    """BWSTest (Baumgartner-Weiss-Schindler test) [baumgartner1998nonparametric]_ detector.

    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallbackBatch, list[BaseCallbackBatch]]]

    :Note:
    - Passing additional arguments to `scipy.stats.bws_test <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bws_test.html>`__ can be done using :func:`compare` kwargs.

    :References:

    .. [baumgartner1998nonparametric] Baumgartner, W., P. Weiß, and H. Schindler.
        "A nonparametric test for the general two-sample problem."
        Biometrics (1998): 1129-1135.

    :Example:

    >>> from frouros.detectors.data_drift import BWSTest
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> X = np.random.normal(loc=0, scale=1, size=100)
    >>> Y = np.random.normal(loc=1, scale=1, size=100)
    >>> detector = BWSTest()
    >>> _ = detector.fit(X=X)
    >>> detector.compare(X=Y)[0]
    StatisticalResult(statistic=29.942072035675395, p_value=0.0001)
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
        test = bws_test(
            x=X_ref,
            y=X,
            alternative=kwargs.get("alternative", "two-sided"),
            method=kwargs.get("method"),
        )
        test = StatisticalResult(
            statistic=test.statistic,
            p_value=test.pvalue,
        )
        return test
