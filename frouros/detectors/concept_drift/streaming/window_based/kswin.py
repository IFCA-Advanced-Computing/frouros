"""KSWIN (Kolmogorov-Smirnov Windowing) module."""

import itertools
from collections import deque
from typing import Any, Optional, Union

import numpy as np
from scipy.stats import ks_2samp

from frouros.callbacks.streaming.base import BaseCallbackStreaming
from frouros.detectors.concept_drift.streaming.window_based.base import (
    BaseWindow,
    BaseWindowConfig,
)


class KSWINConfig(BaseWindowConfig):
    """KSWIN (Kolmogorov-Smirnov Windowing) [raab2020reactive]_ configuration.

    :param alpha: significance value, defaults to 0.0001
    :type alpha: float
    :param seed: seed value, defaults to None
    :type seed: Optional[int]
    :param min_num_instances: minimum numbers of instances to start looking for changes, defaults to 100
    :type min_num_instances: int
    :param num_test_instances: numbers of instances to be used by the statistical test, defaults to 30
    :type num_test_instances: int
    :raises ValueError: Value error exception if seed is not valid

    :References:

    .. [raab2020reactive] Raab, Christoph, Moritz Heusinger, and Frank-Michael Schleif.
        "Reactive soft prototype computing for concept drift streams."
        Neurocomputing 416 (2020): 340-351.
    """  # noqa: E501  # pylint: disable=line-too-long

    def __init__(  # noqa: D107
        self,
        alpha: float = 0.0001,
        seed: Optional[int] = None,
        min_num_instances: int = 100,
        num_test_instances: int = 30,
    ) -> None:
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


class KSWIN(BaseWindow):
    """KSWIN (Kolmogorov-Smirnov Windowing) [raab2020reactive]_ detector.

    :param config: configuration object of the detector, defaults to None. If None, the default configuration of :class:`KSWINConfig` is used.
    :type config: Optional[KSWINConfig]
    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallbackStreaming, list[BaseCallbackStreaming]]]

    :References:

    .. [raab2020reactive] Raab, Christoph, Moritz Heusinger, and Frank-Michael Schleif.
        "Reactive soft prototype computing for concept drift streams."
        Neurocomputing 416 (2020): 340-351.

    :Example:

    >>> from frouros.detectors.concept_drift import KSWIN, KSWINConfig
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> dist_a = np.random.normal(loc=0.2, scale=0.01, size=1000)
    >>> dist_b = np.random.normal(loc=0.8, scale=0.04, size=1000)
    >>> stream = np.concatenate((dist_a, dist_b))
    >>> detector = KSWIN(config=KSWINConfig(seed=31))
    >>> for i, value in enumerate(stream):
    ...     _ = detector.update(value=value)
    ...     if detector.drift:
    ...         print(f"Change detected at step {i}")
    ...         break
    Change detected at step 1016
    """  # noqa: E501  # pylint: disable=line-too-long

    config_type = KSWINConfig

    def __init__(  # noqa: D107
        self,
        config: Optional[KSWINConfig] = None,
        callbacks: Optional[
            Union[BaseCallbackStreaming, list[BaseCallbackStreaming]]
        ] = None,
    ) -> None:
        super().__init__(
            config=config,
            callbacks=callbacks,
        )
        self.additional_vars = {
            "window": deque(maxlen=self.config.min_num_instances),
        }
        self._set_additional_vars_callback()

    @property
    def window(self) -> deque:  # type: ignore
        """Window queue property.

        :return: window queue
        :rtype: deque
        """
        return self._additional_vars["window"]

    @window.setter
    def window(self, value: deque) -> None:  # type: ignore
        """Window queue setter.

        :param value: value to be set
        :type value: deque
        :raises TypeError: Type error exception
        """
        if not isinstance(value, deque):
            raise TypeError("value must be of type deque.")
        self._additional_vars["window"] = value

    def _update(self, value: Union[int, float], **kwargs: Any) -> None:
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
