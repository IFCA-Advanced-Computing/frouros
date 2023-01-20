"""KSTest (Kolmogorov-Smirnov test) module."""

import numpy as np  # type: ignore
from scipy.stats import ks_2samp  # type: ignore

from frouros.data_drift.base import NumericalData, UnivariateData
from frouros.data_drift.batch.statistical_test.base import (
    StatisticalTestBase,
    StatisticalResult,
)


class KSTest(StatisticalTestBase):
    """KSTest (Kolmogorov-Smirnov test) algorithm class."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__(data_type=NumericalData(), statistical_type=UnivariateData())

    def _statistical_test(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> StatisticalResult:
        test = ks_2samp(
            data1=X_ref_,
            data2=X,
            alternative=kwargs.get("alternative", "two-sided"),
            mode=kwargs.get("method", "auto"),
        )
        test = StatisticalResult(statistic=test.statistic, p_value=test.pvalue)
        return test
