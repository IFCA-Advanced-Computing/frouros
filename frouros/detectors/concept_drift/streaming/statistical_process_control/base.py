"""Base concept drift SPC (statistical process control) module."""

import abc
from typing import Dict, List, Optional, Tuple, Union

import numpy as np  # type: ignore

from frouros.callbacks.streaming.base import BaseCallbackStreaming
from frouros.detectors.concept_drift.exceptions import InvalidAverageRunLengthError
from frouros.detectors.concept_drift.streaming.base import (
    BaseConceptDriftStreamingConfig,
    BaseConceptDriftStreaming,
)
from frouros.utils.stats import Mean


class BaseSPCConfig(BaseConceptDriftStreamingConfig):
    """Class representing a SPC configuration class."""

    def __init__(
        self,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        min_num_instances: int = 30,
    ) -> None:
        """Init method.

        :param warning_level: warning level factor
        :type warning_level: float
        :param drift_level: drift level factor
        :type drift_level: float
        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        """
        super().__init__(min_num_instances=min_num_instances)
        self.warning_level = warning_level
        self.drift_level = drift_level

    @property
    def drift_level(self) -> float:
        """Drift level property.

        :return: drift level to use in detecting drift
        :rtype: float
        """
        return self._drift_level  # type: ignore

    @drift_level.setter
    def drift_level(self, value: float) -> None:
        """Drift level setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value <= 0.0:
            raise ValueError("drift level must be greater than 0.0.")
        if value <= self.warning_level:
            raise ValueError("drift level must be greater than warning level.")
        self._drift_level = value

    @property
    def warning_level(self) -> float:
        """Warning level property.

        :return: warning level to use in detecting drift
        :rtype: float
        """
        return self._warning_level

    @warning_level.setter
    def warning_level(self, value: float) -> None:
        """Warning level setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value <= 0.0:
            raise ValueError("warning level must be greater than 0.0.")
        self._warning_level = value


class BaseSPC(BaseConceptDriftStreaming):
    """Abstract class representing an SPC estimator."""

    config_type = BaseSPCConfig

    def __init__(
        self,
        config: Optional[BaseSPCConfig] = None,
        callbacks: Optional[
            Union[BaseCallbackStreaming, List[BaseCallbackStreaming]]
        ] = None,
    ) -> None:
        """Init method.

        :param config: configuration parameters
        :type config: Optional[BaseSPCConfig]
        :param callbacks: callbacks
        :type callbacks: Optional[Union[BaseCallbackStreaming,
        List[BaseCallbackStreaming]]]
        """
        super().__init__(
            config=config,
            callbacks=callbacks,
        )
        self.additional_vars = {
            "warning": False,
        }
        self._set_additional_vars_callback()

    @property
    def warning(self) -> bool:
        """Warning property.

        :return: warning value
        :rtype: bool
        """
        return self._additional_vars["warning"]

    @warning.setter
    def warning(self, value: bool) -> None:
        """Warning setter.

        :param value: value to be set
        :type value: bool
        """
        self._additional_vars["warning"] = value

    def reset(self) -> None:
        """Reset method."""
        super().reset()
        self.warning = False

    @property
    def status(self) -> Dict[str, bool]:
        """Status property.

        :return: status dict
        :rtype: Dict[str, bool]
        """
        return {**super().status, "warning": self.warning}

    @abc.abstractmethod
    def _update(self, value: Union[int, float], **kwargs) -> None:
        pass


class BaseSPCError(BaseSPC):
    """Abstract class representing a SPC error estimator."""

    config_type = BaseSPCConfig

    def __init__(
        self,
        config: Optional[BaseSPCConfig] = None,
        callbacks: Optional[
            Union[BaseCallbackStreaming, List[BaseCallbackStreaming]]
        ] = None,
    ) -> None:
        """Init method.

        :param config: configuration parameters
        :type config: Optional[BaseSPCConfig]
        :param callbacks: callbacks
        :type callbacks: Optional[Union[BaseCallbackStreaming,
        List[BaseCallbackStreaming]]]
        """
        super().__init__(
            config=config,
            callbacks=callbacks,
        )
        self.additional_vars = {
            "error_rate": Mean(),
            "min_error_rate": float("inf"),
            "min_std": float("inf"),
            **self.additional_vars,  # type: ignore
        }
        self._set_additional_vars_callback()

    @property
    def error_rate(self) -> Mean:
        """Error rate property.

        :return: error rate to use
        :rtype: Mean
        """
        return self._additional_vars["error_rate"]

    @error_rate.setter
    def error_rate(self, value: Mean) -> None:
        """Error rate setter.

        :param value: value to be set
        :type value: Mean
        """
        self._additional_vars["error_rate"] = value

    @property
    def min_error_rate(self) -> float:
        """Minimum error rate property.

        :return: minimum error rate to use
        :rtype: float
        """
        return self._additional_vars["min_error_rate"]

    @min_error_rate.setter
    def min_error_rate(self, value: float) -> None:
        """Minimum error rate setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("min_error_rate must be great or equal than 0.")
        self._additional_vars["min_error_rate"] = value

    @property
    def min_error_rate_plus_std(self) -> float:
        """Minimum error rate + std property.

        :return: minimum error rate + std to determine if a change is happening
        :rtype: float
        """
        return self.min_error_rate + self.min_std

    @property
    def min_std(self) -> float:
        """Minimum standard deviation property.

        :return: minimum standard deviation to use
        :rtype: float
        """
        return self._additional_vars["min_std"]

    @min_std.setter
    def min_std(self, value: float) -> None:
        """Minimum standard deviation setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("min_std must be great or equal than 0.")
        self._additional_vars["min_std"] = value

    def _calculate_error_rate_plus_std(self) -> Tuple[float, float]:
        std = np.sqrt(
            self.error_rate.mean * (1 - self.error_rate.mean) / self.num_instances
        )
        error_rate_plus_std = self.error_rate.mean + std
        return error_rate_plus_std, std

    def _update_min_values(self, error_rate_plus_std: float, std: float) -> None:
        if error_rate_plus_std < self.min_error_rate_plus_std:
            self.min_error_rate = self.error_rate.mean
            self.min_std = std

    @staticmethod
    def _check_threshold(
        error_rate_plus_std: float, min_error_rate: float, min_std: float, level: float
    ) -> bool:
        return error_rate_plus_std > min_error_rate + level * min_std

    def reset(self) -> None:
        """Reset method."""
        super().reset()
        self.error_rate = Mean()
        self.min_error_rate = float("inf")
        self.min_std = float("inf")

    @abc.abstractmethod
    def _update(self, value: Union[int, float], **kwargs) -> None:
        pass


class BaseECDDConfig(BaseConceptDriftStreamingConfig):
    """Class representing a ECDD configuration class."""

    average_run_length_map = {
        100: lambda p: 2.76
        - 6.23 * p
        + 18.12 * np.power(p, 3)
        - 312.45 * np.power(p, 5)
        + 1002.18 * np.power(p, 7),
        400: lambda p: 3.97
        - 6.56 * p
        + 48.73 * np.power(p, 3)
        - 330.13 * np.power(p, 5)
        + 848.18 * np.power(p, 7),
        1000: lambda p: 1.17
        + 7.56 * p
        - 21.24 * np.power(p, 3)
        + 112.12 * np.power(p, 5)
        - 987.23 * np.power(p, 7),
    }

    def __init__(
        self,
        lambda_: float = 0.2,
        average_run_length: int = 400,
        warning_level: float = 0.5,
        min_num_instances: int = 30,
    ) -> None:
        """Init method.

        :param average_run_length: expected time between false positive detections
        :type average_run_length: int
        :param lambda_: weight given to recent data compared to older data
        :type lambda_: float
        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        :raises InvalidAverageRunLengthError: Invalid average run length error exception
        """
        super().__init__(min_num_instances=min_num_instances)
        try:
            self.control_limit_func = self.average_run_length_map[average_run_length]
        except KeyError as e:
            raise InvalidAverageRunLengthError(
                "average_run_length must be 100, 400 or 1000."
            ) from e
        self.lambda_ = lambda_
        self.warning_level = warning_level

    @property
    def lambda_(self) -> float:
        """Weight recent data property.

        :return: weight given to recent data
        :rtype: float
        """
        return self._lambda_

    @lambda_.setter
    def lambda_(self, value: float) -> None:
        """Weight recent data setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if not 0.0 <= value <= 1.0:
            raise ValueError("lambda_ must be in the range [0, 1].")
        self._lambda_ = value

    @property
    def warning_level(self) -> float:
        """Warning level property.

        :return: warning level to use in detecting drift
        :rtype: float
        """
        return self._warning_level

    @warning_level.setter
    def warning_level(self, value: float) -> None:
        """Warning level setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if not 0.0 < value < 1.0:
            raise ValueError("warning level must be in the range (0.0, 1.0).")
        self._warning_level = value
