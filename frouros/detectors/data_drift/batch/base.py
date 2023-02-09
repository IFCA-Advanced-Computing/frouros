"""Data drift batch base module."""

import abc

from typing import Any, Dict, Optional, List, Tuple, Union
import numpy as np  # type: ignore

from frouros.callbacks import Callback
from frouros.detectors.data_drift.base import (
    DataDriftBase,
    DataTypeBase,
    StatisticalTypeBase,
)
from frouros.detectors.data_drift.exceptions import (
    DimensionError,
    MismatchDimensionError,
    MissingFitError,
)


class DataDriftBatchBase(DataDriftBase):
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
        super().__init__(
            callbacks=callbacks,
            data_type=data_type,
            statistical_type=statistical_type,
        )
        for callback in self.callbacks:  # type: ignore
            callback.set_detector(detector=self)

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
        self.X_ref = X  # type: ignore
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
                X_ref=self.X_ref,
                X_test=X,
            )

        callbacks_logs = self._get_callbacks_logs()
        return result, callbacks_logs

    def reset(self) -> None:
        """Reset method."""
        self.X_ref = None  # type: ignore

    def _common_checks(self, X: np.ndarray) -> None:  # noqa: N803
        self._check_is_fitted()
        self._check_compare_dimensions(X=X)

    def _check_fit_dimensions(self, X: np.ndarray) -> None:  # noqa: N803
        try:
            if not self.statistical_type.dim_check(X.shape[1], 1):  # type: ignore
                raise DimensionError(f"Dimensions of X ({X.shape[-1]})")
        except IndexError as e:
            if not self.statistical_type.dim_check(X.ndim, 1):  # type: ignore
                raise DimensionError(f"Dimensions of X ({X.ndim})") from e

    def _check_compare_dimensions(self, X: np.ndarray) -> None:  # noqa: N803
        try:
            if self.X_ref.shape[1] != X.shape[1]:  # type: ignore
                raise MismatchDimensionError(
                    f"Dimensions of X_ref ({self.X_ref.shape[-1]}) "  # type: ignore
                    f"and X ({X.shape[-1]}) must be equal"
                )
        except IndexError as e:
            if self.X_ref.ndim != X.ndim:  # type: ignore
                raise MismatchDimensionError(f"Dimensions of X ({X.ndim})") from e

    def _check_is_fitted(self):
        if self.X_ref is None:
            raise MissingFitError("fit method has not been called")

    def _specific_checks(self, X: np.ndarray) -> None:  # noqa: N803
        pass

    @abc.abstractmethod
    def _apply_method(
        self, X_ref: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Any:
        pass

    def _get_result(
        self, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Union[List[float], List[Tuple[float, float]], Tuple[float, float]]:
        result = self._apply_method(  # type: ignore # pylint: disable=not-callable
            X_ref=self.X_ref, X=X, **kwargs
        )
        return result
