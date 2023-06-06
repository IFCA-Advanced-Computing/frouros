"""KSTest (Kolmogorov-Smirnov test) module."""

from typing import Optional, List, Union

import numpy as np  # type: ignore
from scipy.stats import ks_2samp  # type: ignore

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.base import NumericalData, UnivariateData
from frouros.detectors.data_drift.batch.statistical_test.base import (
    BaseStatisticalTest,
    StatisticalResult,
)


class KSTest(BaseStatisticalTest):
    """KSTest (Kolmogorov-Smirnov test) [massey1951kolmogorov]_ detector.

    :References:

    .. [massey1951kolmogorov] Massey Jr, Frank J.
        "The Kolmogorov-Smirnov test for goodness of fit."
        Journal of the American statistical Association 46.253 (1951): 68-78.
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
        test = ks_2samp(
            data1=X_ref,
            data2=X,
            alternative=kwargs.get("alternative", "two-sided"),
            method=kwargs.get("method", "auto"),
        )
        test = StatisticalResult(statistic=test.statistic, p_value=test.pvalue)
        return test
