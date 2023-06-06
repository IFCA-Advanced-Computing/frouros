"""STEPD (Statistical test of equal proportions) module."""

from typing import List, Optional, Union

import numpy as np  # type: ignore
from scipy.stats import norm  # type: ignore

from frouros.callbacks.streaming.base import BaseCallbackStreaming
from frouros.detectors.concept_drift.streaming.window_based.base import (
    BaseWindowConfig,
    BaseWindow,
)
from frouros.utils.data_structures import AccuracyQueue


class STEPDConfig(BaseWindowConfig):
    """STEPD (Statistical test of equal proportions) [nishida2007detecting]_ configuration.

    :References:

    .. [nishida2007detecting] Nishida, Kyosuke, and Koichiro Yamauchi.
        "Detecting concept drift using statistical testing." Discovery science.
        Vol. 4755. 2007.
    """

    def __init__(
        self,
        alpha_d: float = 0.003,
        alpha_w: float = 0.05,
        min_num_instances: int = 30,
    ) -> None:
        """Init method.

        :param alpha_d: significance value for overall
        :type alpha_d: float
        :param alpha_w: significance value for last
        :type alpha_w: float
        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        """
        super().__init__(min_num_instances=min_num_instances)
        self.alpha_d = alpha_d
        self.alpha_w = alpha_w

    @property
    def alpha_d(self) -> float:
        """Significance level d property.

        :return: significance level d
        :rtype: float
        """
        return self._alpha_d

    @alpha_d.setter
    def alpha_d(self, value: float) -> None:
        """Significance level d setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value <= 0.0:
            raise ValueError("alpha_d must be greater than 0.0.")
        self._alpha_d = value

    @property
    def alpha_w(self) -> float:
        """Significance level w property.

        :return: significance level w
        :rtype: float
        """
        return self._alpha_w

    @alpha_w.setter
    def alpha_w(self, value: float) -> None:
        """Significance level w setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value <= 0.0:
            raise ValueError("alpha_w must be greater than 0.0.")
        if value <= self.alpha_d:
            raise ValueError("alpha_w must be greater than alpha_d.")
        self._alpha_w = value


class STEPD(BaseWindow):
    """STEPD (Statistical test of equal proportions) [nishida2007detecting]_ detector.

    :References:

    .. [nishida2007detecting] Nishida, Kyosuke, and Koichiro Yamauchi.
        "Detecting concept drift using statistical testing." Discovery science.
        Vol. 4755. 2007.
    """

    config_type = STEPDConfig  # type: ignore

    def __init__(
        self,
        config: Optional[STEPDConfig] = None,
        callbacks: Optional[
            Union[BaseCallbackStreaming, List[BaseCallbackStreaming]]
        ] = None,
    ) -> None:
        """Init method.

        :param config: configuration parameters
        :type config: Optional[STEPDConfig]
        :param callbacks: callbacks
        :type callbacks: Optional[Union[BaseCallbackStreaming,
        List[BaseCallbackStreaming]]]
        """
        super().__init__(
            config=config,
            callbacks=callbacks,
        )
        self.additional_vars = {
            "correct_total": 0,
            # FIXME include get method in AccuracyQueue  # pylint: disable=fixme
            "window_accuracy": AccuracyQueue(max_len=self.config.min_num_instances),
            **self.additional_vars,  # type: ignore
        }
        self.warning = False
        self._set_additional_vars_callback()
        self._min_num_instances = 2 * self.config.min_num_instances
        self._distribution = norm()

    @property
    def warning(self) -> bool:
        """Warning property.

        :return: warning
        :rtype: bool
        """
        return self._warning

    @warning.setter
    def warning(self, value: bool) -> None:
        """Warning setter.

        :param value: value to be set
        :type value: bool
        """
        self._warning = value

    @property
    def correct_total(self) -> int:
        """Number of correct labels property.

        :return: accuracy scorer function
        :rtype: int
        """
        return self._additional_vars["correct_total"]

    @correct_total.setter
    def correct_total(self, value: int) -> None:
        """Number of correct labels setter.

        :param value: value to be set
        :type value: int
        """
        self._additional_vars["correct_total"] = value

    @property
    def num_instances_overall(self) -> int:
        """Number of overall instances property.

        :return: number of overall instances
        :rtype: int
        """
        return self.num_instances - self.num_instances_window

    @property
    def correct_overall(self) -> int:
        """Number of correct overall labels property.

        :return: number of overall labels
        :rtype: int
        """
        return self.correct_total - self.correct_window

    @property
    def correct_window(self) -> int:
        """Number of correct window labels property.

        :return: number of window labels
        :rtype: int
        """
        return self.window_accuracy.num_true

    @property
    def num_instances_window(self) -> int:
        """Number of window instances property.

        :return: number of window instances
        :rtype: int
        """
        return self.window_accuracy.size

    @property
    def window_accuracy(self) -> AccuracyQueue:
        """Accuracy window property.

        :return: accuracy window
        :rtype: AccuracyQueue
        """
        return self._additional_vars["window_accuracy"]

    @window_accuracy.setter
    def window_accuracy(self, value: AccuracyQueue) -> None:
        """Accuracy window setter.

        :param value: value to be set
        :type value: AccuracyQueue
        :raises TypeError: Type error exception
        """
        if not isinstance(value, AccuracyQueue):
            raise TypeError("value must be of type AccuracyQueue")
        self._additional_vars["window_accuracy"] = value

    def _calculate_statistic(self):
        p_hat = self.correct_total / self.num_instances
        num_instances_inv = (
            1 / self.num_instances_overall + 1 / self.num_instances_window
        )
        statistic = (
            np.abs(
                self.correct_overall / self.num_instances_overall
                - self.correct_window / self.num_instances_window
            )
            - 0.5 * num_instances_inv
        ) / np.sqrt((p_hat * (1 - p_hat) * num_instances_inv))
        return statistic

    def reset(self) -> None:
        """Reset method."""
        super().reset()
        self.correct_total = 0
        self.window_accuracy.clear()

    def _update(self, value: Union[int, float], **kwargs) -> None:
        self.num_instances += 1

        self.correct_total += np.sum(value)
        self.window_accuracy.enqueue(value=value)

        if self.num_instances >= self._min_num_instances:
            statistic = self._calculate_statistic()
            p_value = self._distribution.sf(np.abs(statistic))  # One-sided test

            if p_value < self.config.alpha_d:  # type: ignore
                # Drift case
                self.drift = True
                self.warning = False
            else:
                if p_value < self.config.alpha_w:  # type: ignore
                    # Warning case
                    self.warning = True
                else:
                    # In-Control
                    self.warning = False
                self.drift = False
        else:
            self.drift, self.warning = False, False
