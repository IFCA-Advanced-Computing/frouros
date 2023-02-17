"""KSWIN (Kolmogorov-Smirnov Windowing) module."""

from collections import deque
import itertools
from typing import List, Optional, Union

from scipy.stats import ks_2samp  # type: ignore
import numpy as np  # type: ignore

from frouros.callbacks import Callback
from frouros.detectors.concept_drift.window_based.base import (
    WindowBaseConfig,
    WindowBased,
)


class KSWINConfig(WindowBaseConfig):
    """KSWIN (Kolmogorov-Smirnov Windowing) configuration class."""

    def __init__(
        self,
        alpha: float = 0.0001,
        seed: Optional[int] = None,
        min_num_instances: int = 100,
        num_test_instances: int = 30,
    ) -> None:
        """Init method.

        :param alpha: significance value
        :type alpha: float
        :param seed: seed value
        :type seed: Optional[int]
        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        :param num_test_instances: numbers of instances
        to be used by the statistical test
        :type num_test_instances: int
        :raises ValueError: Value error exception
        """
        try:
            np.random.seed(seed=seed)
        except ValueError as e:
            raise e
        super().__init__(min_num_instances=min_num_instances)
        self.alpha = alpha
        self.num_test_instances = num_test_instances

    @property
    def alpha(self) -> float:
        """Significance value property.

        :return: significance value
        :rtype: float
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: int) -> None:
        """Significance value setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value <= 0:
            raise ValueError("alpha value must be greater than 0.")
        self._alpha = value

    @property
    def num_test_instances(self) -> int:
        """Number of tests instances property.

        :return: number of tests instances to be used by the statistical test
        :rtype: int
        """
        return self._num_test_instances

    @num_test_instances.setter
    def num_test_instances(self, value: int) -> None:
        """Number of tests instances value setter.

        :param value: value to be set
        :type value: int
        :raises ValueError: Value error exception
        """
        if value > self.min_num_instances:
            raise ValueError(
                "num_test_instances value must be smaller or equal than "
                "min_num_instances."
            )
        if value < 1:
            raise ValueError("num_test_instances value must be greater than 0.")
        self._num_test_instances = value


class KSWIN(WindowBased):
    """KSWIN (Kolmogorov-Smirnov Windowing) algorithm class."""

    config_type = KSWINConfig

    def __init__(
        self,
        config: Optional[KSWINConfig] = None,
        callbacks: Optional[Union[Callback, List[Callback]]] = None,
    ) -> None:
        """Init method.

        :param config: configuration parameters
        :type config: Optional[KSWINConfig]
        :param callbacks: callbacks
        :type callbacks: Optional[Union[Callback, List[Callback]]]
        """
        super().__init__(
            config=config,
            callbacks=callbacks,
        )
        self.additional_vars = {
            "window": deque(maxlen=self.config.min_num_instances),
        }
        self._set_additional_vars_callback()

    @property
    def window(self) -> deque:
        """Window queue property.

        :return: window queue
        :rtype: deque
        """
        return self._additional_vars["window"]

    @window.setter
    def window(self, value: deque) -> None:
        """Window queue setter.

        :param value: value to be set
        :type value: deque
        :raises TypeError: Type error exception
        """
        if not isinstance(value, deque):
            raise TypeError("value must be of type deque.")
        self._additional_vars["window"] = value

    def _update(self, value: Union[int, float], **kwargs) -> None:
        self.num_instances += 1

        self.window.append(value)

        window_size = len(self.window)
        if window_size >= self.config.min_num_instances:
            # fmt: off
            num_first_samples = window_size - self.config.num_test_instances  # type: ignore # noqa: E501 pylint: disable=line-too-long
            r_samples = [*itertools.islice(self.window,
                                           num_first_samples,
                                           window_size)]
            w_samples = np.random.choice(
                a=[*itertools.islice(self.window, 0, num_first_samples)],
                size=self.config.num_test_instances,  # type: ignore
                replace=False,
            )
            # fmt: on
            _, p_value = ks_2samp(
                data1=w_samples,
                data2=r_samples,
                alternative="two-sided",
                method="auto",
            )

            if p_value <= self.config.alpha:  # type: ignore
                # Drift detected
                self.drift = True
            else:
                self.drift = False
        else:
            self.drift = False

    def reset(self) -> None:
        """Reset method."""
        super().reset()
        self.window.clear()