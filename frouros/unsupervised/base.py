"""Unsupervised base module."""

import abc
from collections import namedtuple

from typing import Callable, Optional, List, Tuple, Union
import numpy as np  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from sklearn.utils.validation import check_array, check_is_fitted  # type: ignore

from frouros.unsupervised.exceptions import MismatchDimensionError


TestResult = namedtuple("TestResult", ["statistic", "pvalue"])


class BaseTest(abc.ABC):
    """Abstract class representing a test type."""

    def __init__(self) -> None:
        """Init method."""
        self.apply_method: Optional[Callable] = None

    @abc.abstractmethod
    def get_test(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Union[List[float], List[Tuple[float, float]], Tuple[float, float]]:
        """Obtain test result.

        :param X_ref_: reference data
        :type X_ref_: numpy.ndarray
        :param X: feature data
        :type X: numpy.ndarray
        :return test result
        :rtype: Union[List[float], List[Tuple[float, float]], Tuple[float, float]]
        """


class UnivariateTest(BaseTest):
    """Class representing a univariate test."""

    def get_test(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Union[List[np.float], List[Tuple[float, float]]]:
        """Obtain test result for each feature.

        :param X_ref_: reference data
        :type X_ref_: numpy.ndarray
        :param X: feature data
        :type X: numpy.ndarray
        :return test result
        :rtype: Union[List[numpy.float], List[Tuple[float, float]]]
        """
        tests = []
        for i in range(X.shape[1]):
            test = self.apply_method(  # pylint: disable=not-callable
                X_ref_=X_ref_[:, i], X=X[:, i], **kwargs  # type: ignore
            )  # type: ignore
            tests.append(test)
        return tests


class MultivariateTest(BaseTest):
    """Class representing a multivariate test."""

    def get_test(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Tuple[float, float]:
        """Obtain test result.

        :param X_ref_: reference data
        :type X_ref_: numpy.ndarray
        :param X: feature data
        :type X: numpy.ndarray
        :return test result
        :rtype: Tuple[float, float]
        """
        test = self.apply_method(  # type: ignore # pylint: disable=not-callable
            X_ref_=X_ref_, X=X, **kwargs
        )
        return test


class UnsupervisedBaseEstimator(abc.ABC, BaseEstimator, TransformerMixin):
    """Abstract class representing an unsupervised estimator."""

    def __init__(self, test_type: BaseTest) -> None:
        """Init method.

        :param test_type: type of test to apply
        :type test_type: BaseTest
        """
        self.X_ref_ = None  # type: ignore
        self.test: Optional[
            Union[List[float], List[Tuple[float, float]], Tuple[float, float]]
        ] = None
        test_type.apply_method = self._apply_method
        self.test_type = test_type

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
        self._X_ref_ = check_array(value, dtype=None) if value is not None else value

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
    ) -> np.ndarray:
        """Transform values.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :return transformed feature data
        :rtype: numpy.ndarray
        """
        X = self._common_checks(X=X)  # noqa: N806
        self._specific_checks(X=X)  # noqa: N806
        self.test = self.get_test(X=X, **kwargs)  # type: ignore
        return X

    def _common_checks(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        check_is_fitted(self, attributes="X_ref_")
        X = check_array(X, dtype=None)  # noqa: N806
        self._check_dimensions(X=X)
        return X

    def _check_dimensions(self, X: np.ndarray) -> None:  # noqa: N803
        self.X_ref_: np.ndarray
        if self.X_ref_.shape[1] != X.shape[1]:
            raise MismatchDimensionError(
                f"Dimensions of X_ref ({self.X_ref_.shape[1]}) "
                f"and X ({X.shape[1]}) must be equal"
            )

    def _specific_checks(self, X: np.ndarray) -> None:  # noqa: N803
        pass

    @abc.abstractmethod
    def _apply_method(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Union[Tuple[float, float], float]:
        pass

    def get_test(
        self, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Union[List[float], List[Tuple[float, float]], Tuple[float, float]]:
        """Obtain test result.

        :param X: feature data
        :type X: numpy.ndarray
        :return test result
        :rtype: Union[List[float], List[Tuple[float, float]], Tuple[float, float]]
        """
        test = self.test_type.get_test(X_ref_=self.X_ref_, X=X, **kwargs)
        return test
