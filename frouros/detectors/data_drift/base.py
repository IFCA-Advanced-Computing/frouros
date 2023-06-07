"""Base data drift module."""


import abc
import operator
from typing import Any, Dict, List, Optional, Union

import numpy as np  # type: ignore

from frouros.callbacks.base import BaseCallback
from frouros.detectors.base import BaseDetector
from frouros.detectors.data_drift.exceptions import DimensionError, MissingFitError


class BaseResult(abc.ABC):
    """Abstract class representing a result."""


class BaseDataType(abc.ABC):
    """Abstract class representing a data type."""

    @abc.abstractmethod
    def __init__(self) -> None:
        """Init method."""

    def __repr__(self) -> str:
        """Repr method.

        :return: repr value
        :rtype: str
        """
        return (
            f"{self.__class__.__name__}"
            f"({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"
        )


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


class BaseStatisticalType(abc.ABC):
    """Abstract class representing a statistical data type."""

    @abc.abstractmethod
    def __init__(self) -> None:
        """Init method."""

    def __repr__(self) -> str:
        """Repr method.

        :return: repr value
        :rtype: str
        """
        return (
            f"{self.__class__.__name__}"
            f"({', '.join(f'{k[1:]}={v}' for k, v in self.__dict__.items())})"
        )


class UnivariateData(BaseStatisticalType):
    """Class representing a univariate data type."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__()
        self.dim_check = operator.eq


class MultivariateData(BaseStatisticalType):
    """Class representing a multivariate data type."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__()
        self.dim_check = operator.ge


class BaseDataDrift(BaseDetector):
    """Abstract class representing a data drift detector."""

    def __init__(
        self,
        data_type: BaseDataType,
        statistical_type: BaseStatisticalType,
        callbacks: Optional[Union[BaseCallback, List[BaseCallback]]] = None,
    ) -> None:
        """Init method.

        :param data_type: data type
        :type data_type: BaseDataType
        :param statistical_type: statistical type
        :type statistical_type: BaseStatisticalType
        :param callbacks: callbacks
        :type callbacks: Optional[Union[BaseCallback, List[Callback]]]
        """
        super().__init__(callbacks=callbacks)
        self.data_type = data_type
        self.statistical_type = statistical_type
        self.X_ref = None  # type: ignore

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

    @property
    def X_ref(self) -> Optional[np.ndarray]:  # noqa: N802
        """Reference data property.

        :return: reference data
        :rtype: Optional[numpy.ndarray]
        """
        return self._X_ref  # type: ignore # pylint: disable=E1101

    @X_ref.setter  # type: ignore
    def X_ref(self, value: Optional[np.ndarray]) -> None:  # noqa: N802
        """Reference data setter.

        :param value: value to be set
        :type value: Optional[numpy.ndarray]
        """
        if value is not None:
            self._check_array(X=value)
        self._X_ref = value

    def fit(self, X: np.ndarray, **kwargs) -> Dict[str, Any]:  # noqa: N803
        """Fit detector.

        :param X: feature data
        :type X: numpy.ndarray
        :return: callbacks logs
        :rtype: Dict[str, Any]
        """
        self._check_fit_dimensions(X=X)
        for callback in self.callbacks:  # type: ignore
            callback.on_fit_start(
                X=X,
                **kwargs,
            )
        self._fit(X=X, **kwargs)
        for callback in self.callbacks:  # type: ignore
            callback.on_fit_end(
                X=X,
                **kwargs,
            )

        logs = self._get_callbacks_logs()
        return logs

    def reset(self) -> None:
        """Reset method."""
        self.X_ref = None

    def _check_fit_dimensions(self, X: np.ndarray) -> None:  # noqa: N803
        try:
            if not self.statistical_type.dim_check(X.shape[1], 1):  # type: ignore
                raise DimensionError(f"Dimensions of X ({X.shape[-1]})")
        except IndexError as e:
            if not self.statistical_type.dim_check(X.ndim, 1):  # type: ignore
                raise DimensionError(f"Dimensions of X ({X.ndim})") from e

    def _check_is_fitted(self):
        if self.X_ref is None:
            raise MissingFitError("fit method has not been called")

    def _common_checks(self) -> None:  # noqa: N803
        self._check_is_fitted()

    @abc.abstractmethod
    def _fit(self, X: np.ndarray) -> None:  # noqa: N803
        pass

    def __repr__(self) -> str:
        """Repr method.

        :return: repr value
        :rtype: str
        """
        return (
            f"{super().__repr__()[:-1]}, "
            f"data_type={self.data_type}, "
            f"statistical_type={self.statistical_type})"
        )
