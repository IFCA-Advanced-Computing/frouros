"""Data drift batch base module."""

import abc

from typing import Any, Dict, Optional, List, Tuple, Union
import numpy as np  # type: ignore

from frouros.callbacks import Callback
from frouros.detectors.data_drift.base import (
    DataDriftBase,
    DataTypeBase,
    ResultBase,
    StatisticalTypeBase,
)


class DataDriftStreamingBase(DataDriftBase):
    """Abstract class representing a data drift streaming detector."""

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
    ) -> Tuple[Optional[ResultBase], Dict[str, Any]]:
        """Update detector.

        :param value: value to use to update the detector
        :type value: Union[int, float]
        :return: update result
        :rtype: Optional[ResultBase]
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
    def _update(self, value: Union[int, float]) -> Optional[ResultBase]:
        pass
