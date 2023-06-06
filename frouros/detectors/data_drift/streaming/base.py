"""Base data drift batch module."""

import abc
from typing import Any, Dict, Optional, List, Tuple, Union

import numpy as np  # type: ignore

from frouros.callbacks.streaming.base import BaseCallbackStreaming
from frouros.detectors.data_drift.base import (
    BaseDataDrift,
    BaseDataType,
    BaseResult,
    BaseStatisticalType,
)
from frouros.utils.checks import check_callbacks


class BaseDataDriftStreaming(BaseDataDrift):
    """Abstract class representing a data drift streaming detector."""

    def __init__(
        self,
        data_type: BaseDataType,
        statistical_type: BaseStatisticalType,
        callbacks: Optional[
            Union[BaseCallbackStreaming, List[BaseCallbackStreaming]]
        ] = None,
    ) -> None:
        """Init method.

        :param data_type: data type
        :type data_type: BaseDataType
        :param statistical_type: statistical type
        :type statistical_type: BaseStatisticalType
        :param callbacks: callbacks
        :type callbacks: Optional[Union[BaseCallbackStreaming],
        List[BaseCallbackStreaming]]
        """
        check_callbacks(
            callbacks=callbacks,
            expected_cls=BaseCallbackStreaming,  # type: ignore
        )
        super().__init__(
            callbacks=callbacks,  # type: ignore
            data_type=data_type,
            statistical_type=statistical_type,
        )
        self.num_instances = 0
        for callback in self.callbacks:  # type: ignore
            callback.set_detector(detector=self)

    def reset(self) -> None:
        """Reset method."""
        super().reset()
        self.num_instances = 0
        self._reset()

    def update(
        self, value: Union[int, float]
    ) -> Tuple[Optional[BaseResult], Dict[str, Any]]:
        """Update detector.

        :param value: value to use to update the detector
        :type value: Union[int, float]
        :return: update result
        :rtype: Optional[BaseResult]
        """
        self._common_checks()  # noqa: N806
        self._specific_checks(X=value)  # noqa: N806
        self.num_instances += 1

        for callback in self.callbacks:  # type: ignore
            callback.on_update_start(value=value)  # type: ignore
        result = self._update(value=value)
        if result is not None:
            for callback in self.callbacks:  # type: ignore
                callback.on_update_end(  # type: ignore
                    value=result.distance,  # type: ignore
                )

        callbacks_logs = self._get_callbacks_logs()
        return result, callbacks_logs

    def _specific_checks(self, X: np.ndarray) -> None:  # noqa: N803
        pass

    @abc.abstractmethod
    def _fit(self, X: np.ndarray) -> None:  # noqa: N803
        pass

    @abc.abstractmethod
    def _reset(self) -> None:
        pass

    @abc.abstractmethod
    def _update(self, value: Union[int, float]) -> Optional[BaseResult]:
        pass
