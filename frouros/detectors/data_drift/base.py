"""Data drift base module."""


import abc
import operator
from typing import Any, Dict, List, Optional, Union

import numpy as np  # type: ignore

from frouros.callbacks import Callback
from frouros.detectors.base import DetectorBase
from frouros.detectors.data_drift.exceptions import DimensionError, MissingFitError


class ResultBase(abc.ABC):
    """Abstract class representing a result."""


class DataTypeBase(abc.ABC):
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


class CategoricalData(DataTypeBase):
    """Class representing categorical data."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__()
        self.output_type = None


class NumericalData(DataTypeBase):
    """Class representing numerical data."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__()
        self.output_type = np.float32


class StatisticalTypeBase(abc.ABC):
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


class UnivariateData(StatisticalTypeBase):
    """Class representing a univariate data type."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__()
        self.dim_check = operator.eq


class MultivariateData(StatisticalTypeBase):
    """Class representing a multivariate data type."""

    def __init__(self) -> None:
        """Init method."""
        super().__init__()
        self.dim_check = operator.ge


class DataDriftBase(DetectorBase):
    """Abstract class representing a data drift detector."""

    def __init__(
        self,
        data_type: DataTypeBase,
        statistical_type: StatisticalTypeBase,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
    ) -> None:
        """Init method.

        :param data_type: data type
        :type data_type: DataTypeBase
        :param statistical_type: statistical type
        :type statistical_type: StatisticalTypeBase
        :param callbacks: callbacks
        :type callbacks: Optional[Union[Callback, List[Callback]]]
        """
        super().__init__(callbacks=callbacks)
        self.data_type = data_type
        self.statistical_type = statistical_type
        self.X_ref = None  # type: ignore

    @property
    def data_type(self) -> DataTypeBase:
        """Data type property.

        :return: data type
        :rtype: DataTypeBase
        """
        return self._data_type

    @data_type.setter
    def data_type(self, value: DataTypeBase) -> None:
        """Data type setter.

        :param value: value to be set
        :type value: DataTypeBase
        :raises TypeError: Type error exception
        """
        if not isinstance(value, DataTypeBase):
            raise TypeError("value must be of type DataTypeBase.")
        self._data_type = value

    @property
    def statistical_type(self) -> StatisticalTypeBase:
        """Statistical type property.

        :return: statistical type
        :rtype: StatisticalTypeBase
        """
        return self._statistical_type

    @statistical_type.setter
    def statistical_type(self, value: StatisticalTypeBase) -> None:
        """Statistical type setter.

        :param value: value to be set
        :type value: StatisticalTypeBase
        :raises TypeError: Type error exception
        """
        if not isinstance(value, StatisticalTypeBase):
            raise TypeError("value must be of type StatisticalTypeBase.")
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
            callback.on_fit_start()
        self._fit(X=X, **kwargs)
        for callback in self.callbacks:  # type: ignore
            callback.on_fit_end()

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
