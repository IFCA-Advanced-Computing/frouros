"""Data drift batch base module."""

import abc

from typing import Optional, List, Union
import numpy as np  # type: ignore

from frouros.callbacks import Callback
from frouros.detectors.data_drift.base import (
    DataDriftBase,
    DataTypeBase,
    ResultBase,
    StatisticalTypeBase,
)


class StatisticalResult(ResultBase):
    """Statistical result class."""

    def __init__(
        self,
        statistic: Union[int, float],
        p_value: Union[int, float],
    ) -> None:
        """Init method.

        :param statistic: statistic value
        :type statistic: Union[int, float]
        :param p_value: p-value
        :type p_value: Union[int, float]
        """
        super().__init__()
        self.statistic = statistic
        self.p_value = p_value

    @property
    def statistic(self) -> Union[int, float]:
        """Statistic value property.

        :return: statistic value
        :rtype: Union[int, float]
        """
        return self._statistic

    @statistic.setter
    def statistic(self, value: Union[int, float]) -> None:
        """Statistic value setter.

        :param value: value to be set
        :type value: Union[int, float]
        """
        self._statistic = value

    @property
    def p_value(self) -> Union[int, float]:
        """P-value property.

        :return: p-value
        :rtype: Union[int, float]
        """
        return self._p_value

    @p_value.setter
    def p_value(self, value: Union[int, float]) -> None:
        """P-value setter.

        :param value: value to be set
        :type value: Union[int, float]
        """
        self._p_value = value


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

    def update(self, value: Union[int, float]) -> Optional[ResultBase]:
        """Update detector.

        :param value: value to use to update the detector
        :type value: Union[int, float]
        :return: update result
        :rtype: Optional[ResultBase]
        """
        self._common_checks()  # noqa: N806
        self._specific_checks(X=value)  # noqa: N806
        self.num_instances += 1
        return self._update(value=value)

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
