"""CVMTest (Cramér-von Mises test) module."""

from typing import Optional

import numpy as np  # type: ignore
from scipy.stats import cramervonmises_2samp  # type: ignore
from sklearn.utils.validation import check_array  # type: ignore

from frouros.unsupervised.base import NumericalData, UnivariateType
from frouros.unsupervised.exceptions import InsufficientSamplesError
from frouros.unsupervised.statistical_test.base import (
    StatisticalTestBaseEstimator,
    TestResult,
)


class CVMTest(StatisticalTestBaseEstimator):
    """CVMTest (Cramér-von Mises test) algorithm class."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__(data_type=NumericalData(), statistical_type=UnivariateType())

    @StatisticalTestBaseEstimator.X_ref_.setter  # type: ignore[attr-defined]
    def X_ref_(self, value: Optional[np.ndarray]) -> None:  # noqa: N802
        """Reference data setter.

        :param value: value to be set
        :type value: Optional[numpy.ndarray]
        """
        if value is not None:
            self._check_sufficient_samples(X=value)
            self._X_ref_ = check_array(value)  # noqa: N806
        else:
            self._X_ref_ = None  # noqa: N806

    def _specific_checks(self, X: np.ndarray) -> None:  # noqa: N803
        self._check_sufficient_samples(X=X)

    @staticmethod
    def _check_sufficient_samples(X: np.ndarray) -> None:  # noqa: N803
        if X.shape[0] < 2:
            raise InsufficientSamplesError("Number of samples must be at least 2.")

    def _statistical_test(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> TestResult:
        test = cramervonmises_2samp(
            x=X_ref_,
            y=X,
            method=kwargs.get("method", "auto"),
        )
        test = TestResult(statistic=test.statistic, p_value=test.pvalue)
        return test
