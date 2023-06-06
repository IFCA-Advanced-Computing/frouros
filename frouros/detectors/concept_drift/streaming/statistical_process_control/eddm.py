"""EDDM (Early drift detection method) module."""

import copy
from typing import List, Optional, Union

import numpy as np  # type: ignore

from frouros.callbacks.streaming.base import BaseCallbackStreaming
from frouros.detectors.concept_drift.streaming.statistical_process_control.base import (
    BaseSPCConfig,
    BaseSPC,
)


class EDDMConfig(BaseSPCConfig):
    """EDDM (Early drift detection method) [baena2006early]_ configuration.

    :References:

    .. [baena2006early] Baena-Garcıa, Manuel, et al. "Early drift detection method."
        Fourth international workshop on knowledge discovery from data streams.
        Vol. 6. 2006.
    """

    def __init__(
        self,
        alpha: float = 0.95,
        beta: float = 0.9,
        level: float = 2.0,
        min_num_misclassified_instances: int = 30,
    ) -> None:
        """Init method.

        :param alpha: warning zone value
        :type alpha: float
        :param beta: change zone value
        :type beta: float
        :param level: level factor
        :type level: float
        :param min_num_misclassified_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_misclassified_instances: int
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.level = level
        self.min_num_misclassified_instances = min_num_misclassified_instances

    @property
    def alpha(self) -> float:
        """Alpha property.

        :return: warning zone value
        :rtype: float
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Alpha setter.

        :param value: value to be set
        :type value: float
        """
        self._alpha = value

    @property
    def beta(self) -> float:
        """Beta property.

        :return: change zone value
        :rtype: float
        """
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        """Beta setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value <= 0.0:
            raise ValueError("beta must be greater than 0.0.")
        if value >= self.alpha:
            raise ValueError("beta must be less than alpha.")
        self._beta = value

    @property
    def level(self) -> float:
        """Level property.

        :return: Level to use in detecting drift
        :rtype: float
        """
        return self._level

    @level.setter
    def level(self, value: float) -> None:
        """Level setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value <= 0.0:
            raise ValueError("drift level must be greater than 0.0.")
        self._level = value

    @property
    def min_num_misclassified_instances(self) -> int:
        """Minimum number of misclassified instances property.

        :return: minimum number of misclassified instances to use
        :rtype: float
        """
        return self._min_num_misclassified_instances

    @min_num_misclassified_instances.setter
    def min_num_misclassified_instances(self, value: int) -> None:
        """Minimum number of misclassified instances setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError(
                "min_num_misclassified_instances must be greater or equal than 0."
            )
        self._min_num_misclassified_instances = value


class EDDM(BaseSPC):
    """EDDM (Early drift detection method) [baena2006early]_ detector.

    :References:

    .. [baena2006early] Baena-Garcıa, Manuel, et al. "Early drift detection method."
        Fourth international workshop on knowledge discovery from data streams.
        Vol. 6. 2006.
    """

    config_type = EDDMConfig  # type: ignore

    def __init__(
        self,
        config: Optional[EDDMConfig] = None,
        callbacks: Optional[
            Union[BaseCallbackStreaming, List[BaseCallbackStreaming]]
        ] = None,
    ) -> None:
        """Init method.

        :param config: configuration parameters
        :type config: Optional[EDDMConfig]
        :param callbacks: callbacks
        :type callbacks: Optional[Union[BaseCallbackStreaming,
        List[BaseCallbackStreaming]]]
        """
        # mean_distance_error = 0.0
        super().__init__(
            config=config,
            callbacks=callbacks,
        )
        mean_distance_error = 0.0
        self.additional_vars = {
            "last_distance_error": 0.0,
            "max_distance_threshold": float("-inf"),
            "mean_distance_error": mean_distance_error,
            "num_misclassified_instances": 0,
            "old_mean_distance_error": copy.copy(mean_distance_error),
            "std_distance_error": 0.0,
            "variance_distance_error": 0.0,
            **self.additional_vars,  # type: ignore
        }
        self._set_additional_vars_callback()

    @property
    def last_distance_error(self) -> float:
        """Last distance error property.

        :return: last distance error
        :rtype: float
        """
        return self._additional_vars["last_distance_error"]

    @last_distance_error.setter
    def last_distance_error(self, value: float) -> None:
        """Last distance error setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("last_distance_error must be great or equal than 0.")
        self._additional_vars["last_distance_error"] = value

    @property
    def max_distance_threshold(self) -> float:
        """Maximum distance threshold property.

        :return: maximum distance threshold
        :rtype: float
        """
        return self._additional_vars["max_distance_threshold"]

    @max_distance_threshold.setter
    def max_distance_threshold(self, value: float) -> None:
        """Maximum distance threshold setter.

        :param value: value to be set
        :type value: float
        """
        self._additional_vars["max_distance_threshold"] = value

    @property
    def mean_distance_error(self) -> float:
        """Mean distance error property.

        :return: mean distance error
        :rtype: float
        """
        return self._additional_vars["mean_distance_error"]

    @mean_distance_error.setter
    def mean_distance_error(self, value: float) -> None:
        """Mean distance error property.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("mean_distance_error must be great or equal than 0.")
        self._additional_vars["mean_distance_error"] = value

    @property
    def num_misclassified_instances(self) -> int:
        """Minimum number of misclassified instances property.

        :return: minimum number of misclassified instances to use
        :rtype: float
        """
        return self._additional_vars["num_misclassified_instances"]

    @num_misclassified_instances.setter
    def num_misclassified_instances(self, value: int) -> None:
        """Minimum number of misclassified instances setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError(
                "num_misclassified_instances must be greater or equal than 0."
            )
        self._additional_vars["num_misclassified_instances"] = value

    @property
    def old_mean_distance_error(self) -> float:
        """Old mean distance error property.

        :return: old mean distance error
        :rtype: float
        """
        return self._additional_vars["old_mean_distance_error"]

    @old_mean_distance_error.setter
    def old_mean_distance_error(self, value: float) -> None:
        """Old mean distance error setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("old_mean_distance_error must be great or equal than 0.")
        self._additional_vars["old_mean_distance_error"] = value

    @property
    def std_distance_error(self) -> float:
        """Standard deviation distance error property.

        :return: standard deviation distance error
        :rtype: float
        """
        return self._additional_vars["std_distance_error"]

    @std_distance_error.setter
    def std_distance_error(self, value: float) -> None:
        """Standard deviation distance error setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("std_distance_error must be great or equal than 0.")
        self._additional_vars["std_distance_error"] = value

    @property
    def variance_distance_error(self) -> float:
        """Variance distance error property.

        :return: variance deviation distance error
        :rtype: float
        """
        return self._additional_vars["variance_distance_error"]

    @variance_distance_error.setter
    def variance_distance_error(self, value: float) -> None:
        """Variance distance error setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value < 0:
            raise ValueError("variance must be great or equal than 0.")
        self._additional_vars["variance_distance_error"] = value

    def _update(self, value: Union[int, float], **kwargs) -> None:
        self.num_instances += 1

        if value == 1:
            self.num_misclassified_instances += 1

            distance = self.num_instances - self.last_distance_error
            self.old_mean_distance_error = self.mean_distance_error
            self.mean_distance_error += (
                distance - self.mean_distance_error
            ) / self.num_misclassified_instances
            self.variance_distance_error += (distance - self.mean_distance_error) * (
                distance - self.old_mean_distance_error
            )
            self.std_distance_error = (
                np.sqrt(self.variance_distance_error / self.num_misclassified_instances)
                if self.num_misclassified_instances > 0
                else 0.0
            )
            self.last_distance_error = self.num_instances

            if (
                self.num_instances
                >= self.config.min_num_misclassified_instances  # type: ignore
            ):

                distance_threshold = (
                    self.mean_distance_error
                    + self.config.level * self.std_distance_error  # type: ignore
                )
                if distance_threshold > self.max_distance_threshold:
                    self.max_distance_threshold = distance_threshold
                    self.drift, self.warning = False, False
                elif (
                    self.num_misclassified_instances
                    >= self.config.min_num_misclassified_instances  # type: ignore
                ):
                    p = distance_threshold / self.max_distance_threshold
                    if p < self.config.beta:  # type: ignore
                        # Out-of-Control
                        self.drift = True
                        self.warning = False
                    else:
                        if p < self.config.alpha:  # type: ignore
                            # Warning
                            self.warning = True
                        else:
                            self.warning = False
                        self.drift = False
        else:
            self.drift, self.warning = False, False

    def reset(self) -> None:
        """Reset method."""
        super().reset()
        self.last_distance_error = 0.0
        self.max_distance_threshold = float("-inf")
        self.mean_distance_error = 0.0
        self.num_misclassified_instances = 0
        self.old_mean_distance_error = copy.copy(self.mean_distance_error)
        self.std_distance_error = 0.0
        self.variance_distance_error = 0.0
