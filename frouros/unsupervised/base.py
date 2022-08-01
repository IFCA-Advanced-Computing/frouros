"""Unsupervised base module."""

import abc
from collections import namedtuple

from typing import Any, Callable, Dict, Optional, List, Tuple, Union
import numpy as np  # type: ignore
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from sklearn.utils.validation import check_array, check_is_fitted  # type: ignore

from frouros.unsupervised.exceptions import MismatchDimensionError


TestResult = namedtuple("TestResult", ["statistic", "p_value"])


class BaseStatisticalType(abc.ABC):
    """Abstract class representing a statistical type."""

    def __init__(self) -> None:
        """Init method."""
        self._apply_method: Optional[Callable] = None

    @abc.abstractmethod
    def get_result(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Union[List[float], List[Tuple[float, float]], Tuple[float, float]]:
        """Obtain result.

        :param X_ref_: reference data
        :type X_ref_: numpy.ndarray
        :param X: feature data
        :type X: numpy.ndarray
        :return result
        :rtype: Union[List[float], List[Tuple[float, float]], Tuple[float, float]]
        """


class UnivariateType(BaseStatisticalType):
    """Class representing a univariate type."""

    def get_result(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Union[List[float], List[Tuple[float, float]]]:
        """Obtain result for each variable (univariate).

        :param X_ref_: reference data
        :type X_ref_: numpy.ndarray
        :param X: feature data
        :type X: numpy.ndarray
        :return univariate result
        :rtype: Union[List[numpy.float], List[Tuple[float, float]]]
        """
        results = []
        for i in range(X.shape[1]):
            result = self._apply_method(  # pylint: disable=not-callable
                X_ref_=X_ref_[:, i], X=X[:, i], **kwargs  # type: ignore
            )  # type: ignore
            results.append(result)
        return results


class MultivariateType(BaseStatisticalType):
    """Class representing a multivariate type."""

    def get_result(
        self, X_ref_: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Tuple[float, float]:
        """Obtain result for multivariate.

        :param X_ref_: reference data
        :type X_ref_: numpy.ndarray
        :param X: feature data
        :type X: numpy.ndarray
        :return multivariate result
        :rtype: Tuple[float, float]
        """
        result = self._apply_method(  # type: ignore # pylint: disable=not-callable
            X_ref_=X_ref_, X=X, **kwargs
        )
        return result


class BaseDataType(abc.ABC):
    """Abstract class representing a data type."""

    @abc.abstractmethod
    def __init__(self) -> None:
        """Init method."""


class CategoricalData(BaseDataType):
    """Class representing categorical data."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__()
        self.output_type = None


class NumericalData(BaseDataType):
    """Class representing numerical data."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__()
        self.output_type = np.float32


class UnsupervisedBaseEstimator(abc.ABC, BaseEstimator, TransformerMixin):
    """Abstract class representing an unsupervised estimator."""

    def __init__(
        self, data_type: BaseDataType, statistical_type: BaseStatisticalType
    ) -> None:
        """Init method.

        :param data_type: data type
        :type data_type: BaseDataType
        :param statistical_type: statistical type
        :type statistical_type: BaseStatisticalType
        """
        self.X_ref_ = None  # type: ignore
        self.data_type = data_type
        self.statistical_type = statistical_type
        statistical_type._apply_method = self._apply_method

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
        self._X_ref_ = (
            check_array(value, dtype=self.data_type.output_type)  # type: ignore
            if value is not None
            else value
        )

    @property
    def data_type(self) -> BaseDataType:
        """Data type property.

        :return: data type
        :rtype: BaseDataType
        """
        return self._data_type

    @data_type.setter
    def data_type(self, value: BaseDataType) -> None:
        """Data type setter.

        :param value: value to be set
        :type value: BaseDataType
        :raises TypeError: Type error exception
        """
        if not isinstance(value, BaseDataType):
            raise TypeError("value must be of type BaseDataType.")
        self._data_type = value

    @property
    def statistical_type(self) -> BaseStatisticalType:
        """Statistical type property.

        :return: statistical type
        :rtype: BaseStatisticalType
        """
        return self._statistical_type

    @statistical_type.setter
    def statistical_type(self, value: BaseStatisticalType) -> None:
        """Statistical type setter.

        :param value: value to be set
        :type value: BaseStatisticalType
        :raises TypeError: Type error exception
        """
        if not isinstance(value, BaseStatisticalType):
            raise TypeError("value must be of type BaseStatisticalType.")
        self._statistical_type = value

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
        :return: fitted estimator
        :rtype: self
        """
        self.X_ref_ = X  # type: ignore
        return self

    @abc.abstractmethod
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
        :type y: numpy.ndarray
        :return: transformed feature data
        :rtype: numpy.ndarray
        """

    def fit_transform(
        self,
        X: np.ndarray,  # noqa: N803
        y: Optional[np.ndarray] = None,
        **fit_params: Dict[str, Any],
    ) -> np.ndarray:
        """Override fit_transform from TransformerMixin.

        This will avoid to use transform when fit and return reference data instead.

        :param X: feature data
        :type X: numpy.ndarray
        :param y: target data
        :type y: numpy.ndarray
        :param fit_params: dict of additional fit parameters
        :return: reference data
        :rtype: numpy.ndarray
        """
        return self.fit(X=X, **fit_params).X_ref_

    def _common_checks(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        check_is_fitted(self, attributes="X_ref_")
        X = check_array(  # noqa: N806
            X, dtype=self.data_type.output_type  # type: ignore # noqa: N806
        )
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
    ) -> Any:
        pass

    def _get_result(
        self, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Union[List[float], List[Tuple[float, float]], Tuple[float, float]]:
        result = self.statistical_type.get_result(X_ref_=self.X_ref_, X=X, **kwargs)
        return result
