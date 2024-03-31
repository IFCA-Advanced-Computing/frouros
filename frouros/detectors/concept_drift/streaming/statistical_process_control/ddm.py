"""DDM (Drift detection method) module."""

from contextlib import suppress
from typing import Any, Optional, Union

from frouros.callbacks.streaming.base import BaseCallbackStreaming
from frouros.detectors.concept_drift.streaming.statistical_process_control.base import (
    BaseSPCConfig,
    BaseSPCError,
)


class DDMConfig(BaseSPCConfig):
    """DDM (Drift detection method) [gama2004learning]_ configuration.

    :param warning_level: warning level factor, defaults to 2.0
    :type warning_level: float
    :param drift_level: drift level factor, defaults to 3.0
    :type drift_level: float
    :param min_num_instances: minimum numbers of instances to start looking for changes, defaults to 30
    :type min_num_instances: int

    :References:

    .. [gama2004learning] Gama, Joao, et al.
        "Learning with drift detection."
        Advances in Artificial Intelligence–SBIA 2004: 17th Brazilian Symposium on
        Artificial Intelligence, Sao Luis, Maranhao, Brazil, September 29-October 1,
        2004. Proceedings 17. Springer Berlin Heidelberg, 2004.
    """  # noqa: E501  # pylint: disable=line-too-long

    def __init__(  # noqa: D107
        self,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        min_num_instances: int = 30,
    ) -> None:
        super().__init__(
            warning_level=warning_level,
            drift_level=drift_level,
            min_num_instances=min_num_instances,
        )


class DDM(BaseSPCError):
    """DDM (Drift detection method) [gama2004learning]_ detector.

    :param config: configuration object of the detector, defaults to None. If None, the default configuration of :class:`DDMConfig` is used.
    :type config: Optional[DDMConfig]
    :param callbacks: callbacks, defaults to None
    :type callbacks: Optional[Union[BaseCallbackStreaming, list[BaseCallbackStreaming]]]

    :Note:
    :func:`update` method expects to receive a value of 0 if the instance is correctly classified (no error) and 1 otherwise (error).

    :References:

    .. [gama2004learning] Gama, Joao, et al.
        "Learning with drift detection."
        Advances in Artificial Intelligence–SBIA 2004: 17th Brazilian Symposium on
        Artificial Intelligence, Sao Luis, Maranhao, Brazil, September 29-October 1,
        2004. Proceedings 17. Springer Berlin Heidelberg, 2004.

    :Example:

    >>> from frouros.detectors.concept_drift import DDM
    >>> import numpy as np
    >>> np.random.seed(seed=31)
    >>> dist_a = np.random.binomial(n=1, p=0.6, size=1000)
    >>> dist_b = np.random.binomial(n=1, p=0.8, size=1000)
    >>> stream = np.concatenate((dist_a, dist_b))
    >>> detector = DDM()
    >>> warning_flag = False
    >>> for i, value in enumerate(stream):
    ...     _ = detector.update(value=value)
    ...     if detector.drift:
    ...         print(f"Change detected at step {i}")
    ...         break
    ...     if not warning_flag and detector.warning:
    ...         print(f"Warning detected at step {i}")
    ...         warning_flag = True
    Warning detected at step 1049
    Change detected at step 1131
    """  # noqa: E501  # pylint: disable=line-too-long

    config_type = DDMConfig

    def __init__(  # noqa: D107
        self,
        config: Optional[DDMConfig] = None,
        callbacks: Optional[
            Union[BaseCallbackStreaming, list[BaseCallbackStreaming]]
        ] = None,
    ) -> None:
        super().__init__(
            config=config,
            callbacks=callbacks,
        )

    def _update(self, value: Union[int, float], **kwargs: Any) -> None:
        self.num_instances += 1
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
                self.drift = True
                self.warning = False
            else:
                warning_flag = self._check_threshold(
                    error_rate_plus_std=error_rate_plus_std,
                    min_error_rate=self.min_error_rate,
                    min_std=self.min_std,
                    level=self.config.warning_level,  # type: ignore
                )
                if warning_flag:
                    # Warning
                    self.warning = True
                    for callback in self.callbacks:  # type: ignore
                        with suppress(AttributeError):
                            callback.on_warning_detected(**kwargs)  # type: ignore
                else:
                    # In-Control
                    self.warning = False
                self.drift = False
        else:
            self.drift, self.warning = False, False
