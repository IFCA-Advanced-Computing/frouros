"""Data drift batch base module."""

import abc

from typing import Any, Dict, Optional, List, Tuple, Union
import numpy as np  # type: ignore

from frouros.callbacks import Callback
from frouros.data_drift.base import DataTypeBase, StatisticalTypeBase
from frouros.data_drift.exceptions import (
    DimensionError,
    MismatchDimensionError,
    MissingFitError,
)


class DataDriftBatchBase(abc.ABC):
    """Abstract class representing a data drift batch detector."""

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
        :type callbacks: Optional[Union[Callback], List[Callback]]
        """
        self.X_ref_ = None  # type: ignore
        self.data_type = data_type
        self.statistical_type = statistical_type
        self.callbacks = callbacks  # type: ignore
        for callback in self.callbacks:  # type: ignore
            callback.set_detector(detector=self)

    @property
    def callbacks(self) -> Optional[List[Callback]]:
        """Callbacks property.

        :return: callbacks
        :rtype: Optional[List[Callback]]
        """
        return self._callbacks  # type: ignore

    @callbacks.setter
    def callbacks(self, value: Optional[Union[Callback, List[Callback]]]) -> None:
        """Callbacks setter.

        :param value: value to be set
        :type value: Optional[Union[Callback, List[Callback]]]
        """
        if value is not None:
            if value is isinstance(value, Callback):
                self._callbacks = [value]
            elif not all(
                isinstance(callback, Callback) for callback in value  # type: ignore
            ):
                raise TypeError("value must be of type None or a list of Callback.")
            self._callbacks = value  # type: ignore
        else:
            self._callbacks = []

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
        if value is not None:
            self._check_array(X=value)
        self._X_ref_ = value

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

    def fit(
        self,
        X: np.ndarray,  # noqa: N803
    ) -> Dict[str, Any]:
        """Fit detector.

        :param X: feature data
        :type X: numpy.ndarray
        :return: callbacks logs
        :rtype: Dict[str, Any]
        """
        self._check_fit_dimensions(X=X)
        for callback in self.callbacks:  # type: ignore
            callback.on_fit_start()
        self.X_ref_ = X  # type: ignore
        for callback in self.callbacks:  # type: ignore
            callback.on_fit_end()

        logs = self._get_callbacks_logs()
        return logs

    def compare(
        self,
        X: np.ndarray,  # noqa: N803
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compare values.

        :param X: feature data
        :type X: numpy.ndarray
        :return: compare result and callbacks logs
        :rtype: Tuple[np.ndarray, Dict[str, Any]]
        """
        for callback in self.callbacks:  # type: ignore
            callback.on_compare_start()  # type: ignore
        result = self._compare(X=X, **kwargs)
        for callback in self.callbacks:  # type: ignore
            callback.on_compare_end(  # type: ignore
                result=result,
                X_ref=self.X_ref_,
                X_test=X,
            )

        callbacks_logs = self._get_callbacks_logs()
        return result, callbacks_logs

    def reset(self) -> None:
        """Reset method."""
        self.X_ref_ = None  # type: ignore

    @abc.abstractmethod
    def _compare(
        self,
        X: np.ndarray,  # noqa: N803
        **kwargs,
    ) -> np.ndarray:
        pass

    def _common_checks(self, X: np.ndarray) -> None:  # noqa: N803
        self._check_is_fitted()
        self._check_compare_dimensions(X=X)

    @staticmethod
    def _check_array(X: Any) -> None:  # noqa: N803
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array")

    def _check_fit_dimensions(self, X: np.ndarray) -> None:  # noqa: N803
        try:
            if not self.statistical_type.dim_check(X.shape[1], 1):  # type: ignore
                raise DimensionError(f"Dimensions of X ({X.shape[-1]})")
        except IndexError as e:
            if not self.statistical_type.dim_check(X.ndim, 1):  # type: ignore
                raise DimensionError(f"Dimensions of X ({X.ndim})") from e

    def _check_compare_dimensions(self, X: np.ndarray) -> None:  # noqa: N803
        if self.X_ref_.shape[-1] != X.shape[-1]:  # type: ignore
            raise MismatchDimensionError(
                f"Dimensions of X_ref ({self.X_ref_.shape[-1]}) "  # type: ignore
                f"and X ({X.shape[-1]}) must be equal"
            )

    def _check_is_fitted(self):
        if self.X_ref_ is None:
            raise MissingFitError("fit method has not been called")

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
        result = self._apply_method(  # type: ignore # pylint: disable=not-callable
            X_ref_=self.X_ref_, X=X, **kwargs
        )
        return result

    def _get_callbacks_logs(self) -> Dict[str, Any]:
        logs = {
            callback.name: callback.logs for callback in self.callbacks  # type: ignore
        }
        return logs
