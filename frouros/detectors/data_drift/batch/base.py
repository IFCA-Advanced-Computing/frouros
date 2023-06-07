"""Base data drift batch module."""

import abc
from typing import Any, Dict, Optional, List, Tuple, Union

import numpy as np  # type: ignore

from frouros.callbacks.batch.base import BaseCallbackBatch
from frouros.detectors.data_drift.base import (
    BaseDataDrift,
    BaseDataType,
    BaseStatisticalType,
)
from frouros.detectors.data_drift.exceptions import (
    MismatchDimensionError,
)
from frouros.utils.checks import check_callbacks


class BaseDataDriftBatch(BaseDataDrift):
    """Abstract class representing a data drift batch detector."""

    def __init__(
        self,
        data_type: BaseDataType,
        statistical_type: BaseStatisticalType,
        callbacks: Optional[Union[BaseCallbackBatch, List[BaseCallbackBatch]]] = None,
    ) -> None:
        """Init method.

        :param data_type: data type
        :type data_type: BaseDataType
        :param statistical_type: statistical type
        :type statistical_type: BaseStatisticalType
        :param callbacks: callbacks
        :type callbacks: Optional[Union[BaseCallbackBatch], List[BaseCallbackBatch]]
        """
        check_callbacks(
            callbacks=callbacks,
            expected_cls=BaseCallbackBatch,  # type: ignore
        )
        super().__init__(
            callbacks=callbacks,  # type: ignore
            data_type=data_type,
            statistical_type=statistical_type,
        )
        for callback in self.callbacks:  # type: ignore
            callback.set_detector(detector=self)

    def _fit(
        self,
        X: np.ndarray,  # noqa: N803
    ) -> None:
        self.X_ref = X  # type: ignore

    def compare(
        self,
        X: np.ndarray,  # noqa: N803
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compare values.

        :param X: feature data
        :type X: numpy.ndarray
        :return: compare result and callbacks logs
        :rtype: Tuple[numpy.ndarray, Dict[str, Any]]
        """
        for callback in self.callbacks:  # type: ignore
            callback.on_compare_start(  # type: ignore
                X_ref=self.X_ref,
                X_test=X,
            )
        result = self._compare(X=X, **kwargs)
        for callback in self.callbacks:  # type: ignore
            callback.on_compare_end(  # type: ignore
                result=result,
                X_ref=self.X_ref,
                X_test=X,
            )

        callbacks_logs = self._get_callbacks_logs()
        return result, callbacks_logs

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

    def _specific_checks(self, X: np.ndarray) -> None:  # noqa: N803
        self._check_compare_dimensions(X=X)

    @abc.abstractmethod
    def _apply_method(
        self, X_ref: np.ndarray, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Any:
        pass

    @abc.abstractmethod
    def _compare(
        self,
        X: np.ndarray,  # noqa: N803
        **kwargs,
    ) -> np.ndarray:
        pass

    def _get_result(
        self, X: np.ndarray, **kwargs  # noqa: N803
    ) -> Union[List[float], List[Tuple[float, float]], Tuple[float, float]]:
        result = self._apply_method(  # type: ignore # pylint: disable=not-callable
            X_ref=self.X_ref, X=X, **kwargs
        )
        return result
