"""RDDM (Reactive Drift detection method) module."""

from typing import Any, Optional, Union

from frouros.callbacks.streaming.base import BaseCallbackStreaming
from frouros.detectors.concept_drift.streaming.statistical_process_control.base import (
    BaseSPCConfig,
    BaseSPCError,
)
from frouros.utils.data_structures import CircularQueue
from frouros.utils.stats import Mean


class RDDMConfig(BaseSPCConfig):
    """RDDM (Reactive Drift detection method) [barros2017rddm]_ configuration.

    :param warning_level: warning level factor, defaults to 1.773
    :type warning_level: float
    :param drift_level: drift level factor, defaults to 2.258
    :type drift_level: float
    :param max_concept_size: maximum size of a concept, defaults to 40000
    :type max_concept_size: int
    :param min_concept_size: reduced size of a concept, defaults to 7000
    :type min_concept_size: int
    :param max_num_instances_warning: maximum number of instances at warning level, defaults to 1400
    :type max_num_instances_warning: int
    :param min_num_instances: minimum numbers of instances to start looking for changes, defaults to 129
    :type min_num_instances: int

    :References:

    .. [barros2017rddm] Barros, Roberto SM, et al.
        "RDDM: Reactive drift detection method."
        Expert Systems with Applications 90 (2017): 344-355.
    """  # noqa: E501  # pylint: disable=line-too-long

    def __init__(  # noqa: D107
        self,
        warning_level: float = 1.773,
        drift_level: float = 2.258,
        max_concept_size: int = 40000,
        min_concept_size: int = 7000,
        max_num_instances_warning: int = 1400,
        min_num_instances: int = 129,
    ) -> None:
        super().__init__(
            drift_level=drift_level,
            warning_level=warning_level,
            min_num_instances=min_num_instances,
        )
        self.max_concept_size = max_concept_size
        self.min_concept_size = min_concept_size
        self.max_num_instances_warning = max_num_instances_warning

    @property
    def max_concept_size(self) -> int:
        """Maximum size of a concept property.

        :return: maximum size of a concept
        :rtype: int
        """
        return self._max_concept_size

    @max_concept_size.setter
    def max_concept_size(self, value: int) -> None:
        """Maximum size of a concept setter.

        :param value: value to be set
        :type value: int
        """
        self._max_concept_size = value

    @property
    def min_concept_size(self) -> int:
        """Minimum size of a concept property.

        :return: minimum size of a concept
        :rtype: int
        """
        return self._min_concept_size

    @min_concept_size.setter
    def min_concept_size(self, value: int) -> None:
        """Minimum size of a concept setter.

        :param value: value to be set
        :type value: int
        """
        self._min_concept_size = value

    @property
    def max_num_instances_warning(self) -> int:
        """Maximum number of instances at warning level property.

        :return: maximum number of instances at warning level
        :rtype: int
        """
        return self._max_num_instances_warning

    @max_num_instances_warning.setter
    def max_num_instances_warning(self, value: int) -> None:
        """Maximum number of instances at warning level setter.

        :param value: value to be set
        :type value: int
        """
        self._max_num_instances_warning = value


class RDDM(BaseSPCError):
    """RDDM (Reactive Drift detection method) [barros2017rddm]_ detector.

    :param config: configuration object of the detector, defaults to None. If None, the default configuration of :class:`RDDMConfig` is used.
    :type config: Optional[RDDMConfig]
    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallbackStreaming, list[BaseCallbackStreaming]]]

    :Note:
    :func:`update` method expects to receive a value of 0 if the instance is correctly classified (no error) and 1 otherwise (error).

    :References:

    .. [barros2017rddm] Barros, Roberto SM, et al.
        "RDDM: Reactive drift detection method."
        Expert Systems with Applications 90 (2017): 344-355.

    :Example:

    >>> from frouros.detectors.concept_drift import RDDM
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> dist_a = np.random.binomial(n=1, p=0.6, size=1000)
    >>> dist_b = np.random.binomial(n=1, p=0.8, size=1000)
    >>> stream = np.concatenate((dist_a, dist_b))
    >>> detector = RDDM()
    >>> warning_flag = False
    >>> for i, value in enumerate(stream):
    ...     _ = detector.update(value=value)
    ...     if detector.drift:
    ...         print(f"Change detected at step {i}")
    ...         break
    ...     if not warning_flag and detector.warning:
    ...         print(f"Warning detected at step {i}")
    ...         warning_flag = True
    Warning detected at step 1036
    Change detected at step 1066
    """  # noqa: E501  # pylint: disable=line-too-long

    config_type = RDDMConfig  # type: ignore

    def __init__(  # noqa: D107
        self,
        config: Optional[RDDMConfig] = None,
        callbacks: Optional[
            Union[BaseCallbackStreaming, list[BaseCallbackStreaming]]
        ] = None,
    ) -> None:
        super().__init__(
            config=config,
            callbacks=callbacks,
        )
        self.additional_vars = {
            "num_warnings": 0,
            "rddm_drift": False,
            "predictions": CircularQueue(
                max_len=self.config.min_concept_size  # type: ignore
            ),
            **self.additional_vars,  # type: ignore
        }
        self._set_additional_vars_callback()

    @property
    def num_warnings(self) -> int:
        """Number of warnings property.

        :return: number of warnings
        :rtype: int
        """
        return self._additional_vars["num_warnings"]

    @num_warnings.setter
    def num_warnings(self, value: int) -> None:
        """Number of warnings setter.

        :param value: value to be set
        :type value: int
        """
        self._additional_vars["num_warnings"] = value

    @property
    def rddm_drift(self) -> bool:
        """Rddm drift property.

        :return: rddmi drift value
        :rtype: bool
        """
        return self._additional_vars["rddm_drift"]

    @rddm_drift.setter
    def rddm_drift(self, value: bool) -> None:
        """Rddm drift setter.

        :param value: value to be set
        :type value: bool
        """
        self._additional_vars["rddm_drift"] = value

    @property
    def predictions(self) -> CircularQueue:
        """Predictions circular queue property.

        :return: predictions circular queue
        :rtype: CircularQueue
        """
        return self._additional_vars["predictions"]

    @predictions.setter
    def predictions(self, value: CircularQueue) -> None:
        """Predictions circular queue setter.

        :param value: value to be set
        :type value: CircularQueue
        :raises TypeError: Type error exception
        """
        if not isinstance(value, CircularQueue):
            raise TypeError("value must be of type CircularQueue.")
        self._additional_vars["predictions"] = value

    def reset(self) -> None:
        """Reset method."""
        super().reset()
        self.rddm_drift = False

    def _update(  # pylint: disable=too-many-branches
        self,
        value: Union[int, float],
        **kwargs: Any,
    ) -> None:
        self.num_instances += 1

        if self.rddm_drift:
            self._rdd_drift_case()

        self.predictions.enqueue(value=value)
        self.error_rate.update(value=value)

        if self.num_instances >= self.config.min_num_instances:
            error_rate_plus_std, std = self._calculate_error_rate_plus_std()

            self._update_min_values(error_rate_plus_std=error_rate_plus_std, std=std)

            drift_flag = self._check_threshold(
                error_rate_plus_std=error_rate_plus_std,
                min_error_rate=self.min_error_rate,
                min_std=self.min_std,
                level=self.config.drift_level,  # type: ignore
            )

            if drift_flag:
                # Out-of-Control
                self.rddm_drift = True
                self.drift = True
                self.warning = False
                if self.num_warnings == 0:
                    self.predictions.maintain_last_element()
            else:
                warning_flag = self._check_threshold(
                    error_rate_plus_std=error_rate_plus_std,
                    min_error_rate=self.min_error_rate,
                    min_std=self.min_std,
                    level=self.config.warning_level,  # type: ignore
                )
                if warning_flag:
                    if (
                        self.num_warnings >= self.config.max_num_instances_warning  # type: ignore # noqa: E501
                    ):
                        self.rddm_drift = True
                        self.drift = True
                        self.predictions.maintain_last_element()
                    else:
                        # Warning
                        self.warning = True
                        self.num_warnings += 1
                        self.drift = False
                else:
                    # In-Control
                    self.drift = False
                    self.warning = False
                    self.num_warnings = 0
                if (
                    self.num_instances >= self.config.max_concept_size  # type: ignore
                    and not self.warning
                ):
                    self.rddm_drift = True
        else:
            self.drift, self.warning = False, False

    def _rdd_drift_case(self) -> None:
        self._reset_stats()
        pos = self.predictions.first
        for _ in range(self.predictions.count):
            self.num_instances += 1
            self.error_rate.update(value=self.predictions[pos])
            error_rate_plus_std, std = self._calculate_error_rate_plus_std()

            if (
                self.drift
                and self.num_instances >= self.config.min_num_instances
                and error_rate_plus_std < self.min_error_rate_plus_std
            ):
                self.min_error_rate = self.error_rate.mean
                self.min_std = std

            pos = (pos + 1) % self.config.min_concept_size  # type: ignore
        self.rddm_drift = False
        self.drift = False

    def _reset_stats(self) -> None:
        self.error_rate = Mean()
        self.min_error_rate = float("inf")
        self.min_std = float("inf")
        self.num_warnings = 0
        self.num_instances = 0
        self.rddm_drift = False
