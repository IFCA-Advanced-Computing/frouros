"""Unsupervised base module."""

import abc

from typing import Any, Optional, List, Tuple
import numpy as np  # type: ignore
from sklearn.utils.validation import check_array, check_is_fitted  # type: ignore

from frouros.unsupervised.statistical_test.exceptions import MismatchDimensionError


class StatisticalTestEstimator(abc.ABC):
    """Abstract class representing a test estimator."""

    def __init__(self) -> None:
        """Init method."""
        self.X_ref_ = None  # type: ignore
        self.test = None

    @property
    def X_ref_(self) -> Optional[np.ndarray]:  # noqa: N802
        """Reference data property.

        :return: reference data
        :rtype: Optional[numpy.ndarray]
        """
        return self._X_ref_  # type: ignore # pylint: disable=E1101

    @X_ref_.setter  # type: ignore
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
        self.X_ref_ = X  # type: ignore
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
        X = self._common_checks(X=X)  # noqa: N806
        self._specific_checks(X=X)  # noqa: N806
        self.test = self._get_test(X=X, **kwargs)
        return X

    def _get_test(
        self, X: np.ndarray, **kwargs  # noqa: N803
    ) -> List[Tuple[float, float]]:
        tests = []
        for i in range(X.shape[1]):
            test = self._statistical_test(
                X_ref_=self.X_ref_[:, i], X=X[:, i], **kwargs  # type: ignore
            )  # type: ignore
            tests.append((test.statistic, test.pvalue))
        return tests

    def _common_checks(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        check_is_fitted(self, attributes="X_ref_")
        X = check_array(X)  # noqa: N806
        self._check_dimensions(X=X)
        return X

    def _check_dimensions(self, X: np.ndarray) -> None:  # noqa: N803
        self.X_ref_: np.ndarray
        if self.X_ref_.shape[1] != X.shape[1]:
            raise MismatchDimensionError(
                f"Dimensions of X_ref ({self.X_ref_.shape[1]}) "
                f"and X ({X.shape[1]}) must be equal"
            )

    # @abc.abstractmethod
    def _specific_checks(self, X: np.ndarray) -> None:  # noqa: N803
        pass

    @staticmethod
    @abc.abstractmethod
    def _statistical_test(
        X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Any:
        pass
