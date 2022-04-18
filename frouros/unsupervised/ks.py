"""KSTest (Kolmogorov-Smirnov test) module."""
from typing import List, Optional, Tuple

import numpy as np  # type: ignore
from scipy.stats import ks_2samp  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from sklearn.utils.validation import check_array, check_is_fitted  # type: ignore

from frouros.unsupervised.exceptions import MisMatchDimensionError


class KSTest(BaseEstimator, TransformerMixin):
    """KSTest (Kolmogorov-Smirnov test) algorithm class."""

    def __init__(self) -> None:
        """Init method."""
        self.X_ref_ = None
        self.test = None

    @property
    def X_ref_(self) -> Optional[np.ndarray]:  # noqa: N802
        """Reference data property.

        :return: reference data
        :rtype: Optional[numpy.ndarray]
        """
        return self._X_ref_  # pylint: disable=E1101

    @X_ref_.setter
    def X_ref_(self, value: Optional[np.ndarray]) -> None:  # noqa: N802
        """Reference data setter.

        :param value: value to be set
        :type value: Optional[numpy.ndarray]
        """
        self._X_ref_ = check_array(value) if value is not None else value  # noqa: N806

    @property
    def test(self) -> Optional[List[Tuple[float, float]]]:
        """Test results property.

        :return: test results
        :rtype: Optional[List[Tuple[float, float]]]
        """
        return self._test

    @test.setter
    def test(self, value: Optional[List[Tuple[float, float]]]) -> None:
        """Test results setter.

        :param value: value to be set
        :type value: Optional[List[Tuple[float, float]]]
        """
        self._test = value

    def fit(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray = None,  # pylint: disable=W0613
    ):
        """Fit estimator.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :type y: numpy.ndarray
        :return fitted estimator
        :rtype: self
        """
        self.X_ref_ = X
        return self

    def transform(
        self,
        X: np.ndarray,  # noqa: N803
        y: np.ndarray = None,  # pylint: disable=W0613
        **kwargs,
    ):
        """Transform values.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :rtype: numpy.ndarray
        """
        check_is_fitted(self, attributes="X_ref_")
        X = check_array(X)  # noqa: N806
        self.X_ref_: np.ndarray
        if self.X_ref_.shape[1] != X.shape[1]:
            raise MisMatchDimensionError(
                f"Dimensions of X_ref ({self.X_ref_.shape[1]}) "
                f"and X ({X.shape[1]}) must be equal"
            )
        tests = []
        for i in range(X.shape[1]):
            test = ks_2samp(
                data1=self.X_ref_[:, i],
                data2=X[:, i],
                alternative=kwargs.get("alternative", "two-sided"),
                mode=kwargs.get("mode", "auto"),
            )
            tests.append((test.statistic, test.pvalue))
        self.test = tests
        return X
