"""Anderson-Darling test module."""

from typing import Optional, List, Union

import numpy as np  # type: ignore
from scipy.stats import anderson_ksamp  # type: ignore

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.base import NumericalData, UnivariateData
from frouros.detectors.data_drift.batch.statistical_test.base import (
    BaseStatisticalTest,
    StatisticalResult,
)


class AndersonDarlingTest(BaseStatisticalTest):
    """Anderson-Darling test [scholz1987k]_ detector.

    :Note:
     p-values are bounded between 0.001 and 0.25 according to scipy documentation [1]_.

    :References:

    .. [scholz1987k] Scholz, Fritz W., and Michael A. Stephens.
        "K-sample Anderson–Darling tests."
        Journal of the American Statistical Association 82.399 (1987): 918-924.
       [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson_ksamp.html  # noqa: E501 # pylint: disable=line-too-long
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
