"""ECDD (EWMA for Concept Drift Detection) module."""

from typing import Any, Optional, Union

import numpy as np

from frouros.callbacks.streaming.base import BaseCallbackStreaming
from frouros.detectors.concept_drift.streaming.statistical_process_control.base import (
    BaseECDDConfig,
    BaseSPC,
)
from frouros.utils.stats import EWMA, Mean


class ECDDWTConfig(BaseECDDConfig):
    """ECDDWT (EWMA Concept Drift Detection Warning) [ross2012exponentially]_ configuration.

    :param lambda_: weight given to recent data compared to older data, defaults to 0.2
    :type lambda_: float
    :param average_run_length: expected time between false positive detections [100, 400 or 1000], defaults to 400
    :type average_run_length: int
    :param warning_level: warning level value, defaults to 0.5
    :type warning_level: float
    :param min_num_instances: minimum numbers of instances to start looking for changes, defaults to 30
    :type min_num_instances: int

    :References:

    .. [ross2012exponentially] Ross, Gordon J., et al.
        "Exponentially weighted moving average charts for detecting concept drift."
        Pattern recognition letters 33.2 (2012): 191-198.
    """  # noqa: E501  # pylint: disable=line-too-long

    def __init__(  # noqa: D107
        self,
        lambda_: float = 0.2,
        average_run_length: int = 400,
        warning_level: float = 0.5,
        min_num_instances: int = 30,
    ) -> None:
        super().__init__(
            lambda_=lambda_,
            average_run_length=average_run_length,
            warning_level=warning_level,
            min_num_instances=min_num_instances,
        )


class ECDDWT(BaseSPC):
    """ECDDWT (EWMA Concept Drift Detection Warning) [ross2012exponentially]_ detector.

    :param config: configuration object of the detector, defaults to None. If None, the default configuration of :class:`ECDDWTConfig` is used.
    :type config: Optional[ECDDWTConfig]
    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallbackStreaming, list[BaseCallbackStreaming]]]

    :Note:
    :func:`update` method expects to receive a value of 0 if the instance is correctly classified (no error) and 1 otherwise (error).

    :References:

    .. [ross2012exponentially] Ross, Gordon J., et al.
        "Exponentially weighted moving average charts for detecting concept drift."
        Pattern recognition letters 33.2 (2012): 191-198.

    :Example:

    >>> from frouros.detectors.concept_drift import ECDDWT
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> dist_a = np.random.binomial(n=1, p=0.6, size=1000)
    >>> dist_b = np.random.binomial(n=1, p=0.8, size=1000)
    >>> stream = np.concatenate((dist_a, dist_b))
    >>> detector = ECDDWT()
    >>> warning_flag = False
    >>> for i, value in enumerate(stream):
    ...     _ = detector.update(value=value)
    ...     if detector.drift:
    ...         print(f"Change detected at step {i}")
    ...         break
    ...     if not warning_flag and detector.warning:
    ...         print(f"Warning detected at step {i}")
    ...         warning_flag = True
    """  # noqa: E501 # pylint: disable=line-too-long

    config_type = ECDDWTConfig  # type: ignore

    def __init__(  # noqa: D107
        self,
        config: Optional[ECDDWTConfig] = None,
        callbacks: Optional[
            Union[BaseCallbackStreaming, list[BaseCallbackStreaming]]
        ] = None,
    ) -> None:
        super().__init__(
            config=config,  # type: ignore
            callbacks=callbacks,
        )
        self.additional_vars = {
            "p": Mean(),
            "z": EWMA(alpha=self.config.lambda_),  # type: ignore
            **self.additional_vars,  # type: ignore
        }
        self._set_additional_vars_callback()
        self._lambda_div_two_minus_lambda = self.config.lambda_ / (  # type: ignore
            2 - self.config.lambda_  # type: ignore
        )

    @property
    def p(self) -> Mean:
        """P property.

        :return: p value
        :rtype: Mean
        """
        return self._additional_vars["p"]

    @p.setter
    def p(self, value: Mean) -> None:
        """P setter.

        :param value: value to be set
        :type value: Mean
        :raises TypeError: Type error exception
        """
        if not isinstance(value, Mean):
            raise TypeError("value must be of type Mean.")
        self._additional_vars["p"] = value

    @property
    def z(self) -> EWMA:
        """Z property.

        :return: z value
        :rtype: Mean
        """
        return self._additional_vars["z"]

    @z.setter
    def z(self, value: EWMA) -> None:
        """Z setter.

        :param value: value to be set
        :type value: EWMA
        :raises TypeError: Type error exception
        """
        if not isinstance(value, EWMA):
            raise TypeError("value must be of type EWMA.")
        self._additional_vars["z"] = value

    def _check_threshold(
        self, control_limit: float, z_variance: float, warning_level: float = 1.0
    ) -> bool:
        return self.z.mean > self.p.mean + warning_level * control_limit * z_variance

    def reset(self) -> None:
        """Reset method."""
        super().reset()
        self.p = Mean()
        self.z = EWMA(alpha=self.config.lambda_)  # type: ignore

    def _update(self, value: Union[int, float], **kwargs: Any) -> None:
        self.num_instances += 1

        self.p.update(value=value)
        self.z.update(value=value)

        if self.num_instances >= self.config.min_num_instances:
            error_rate_variance = self.p.mean * (1 - self.p.mean)
            z_variance = np.sqrt(
                self._lambda_div_two_minus_lambda
                * (1 - self.z.one_minus_alpha ** (2 * self.num_instances))
                * error_rate_variance
            )
            control_limit = self.config.control_limit_func(  # type: ignore
                p=self.p.mean,
            )

            drift_flag = self._check_threshold(
                control_limit=control_limit,
                z_variance=z_variance,
            )

            if drift_flag:
                # Out-of-Control
                self.drift = True
                self.warning = False
            else:
                warning_flag = self._check_threshold(
                    control_limit=control_limit,
                    z_variance=z_variance,
                    warning_level=self.config.warning_level,  # type: ignore
                )
                if warning_flag:
                    # Warning
                    self.warning = True
                else:
                    # In-Control
                    self.warning = False
                self.drift = False
        else:
            self.drift, self.warning = False, False
