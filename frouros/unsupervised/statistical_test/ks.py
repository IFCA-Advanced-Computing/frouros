"""KSTest (Kolmogorov-Smirnov test) module."""

from typing import Tuple

import numpy as np  # type: ignore
from scipy.stats import ks_2samp  # type: ignore

from frouros.unsupervised.base import UnivariateTest
from frouros.unsupervised.statistical_test.base import StatisticalTestBaseEstimator


class KSTest(StatisticalTestBaseEstimator):
    """KSTest (Kolmogorov-Smirnov test) algorithm class."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__(test_type=UnivariateTest())

    @staticmethod
    def _statistical_test(
        X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Tuple[float, float]:
        test = ks_2samp(
            data1=X_ref_,
            data2=X,
            alternative=kwargs.get("alternative", "two-sided"),
            mode=kwargs.get("method", "auto"),
        )
        return test.statistic, test.pvalue
