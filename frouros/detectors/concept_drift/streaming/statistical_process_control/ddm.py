"""DDM (Drift detection method) module."""

from contextlib import suppress
from typing import Union

from frouros.detectors.concept_drift.streaming.statistical_process_control.base import (
    BaseSPCConfig,
    BaseSPCError,
)


class DDMConfig(BaseSPCConfig):
    """DDM (Drift detection method) [gama2004learning]_ configuration.

    :References:

    .. [gama2004learning] Gama, Joao, et al.
        "Learning with drift detection."
        Advances in Artificial Intelligence–SBIA 2004: 17th Brazilian Symposium on
        Artificial Intelligence, Sao Luis, Maranhao, Brazil, September 29-Ocotber 1,
        2004. Proceedings 17. Springer Berlin Heidelberg, 2004.
    """


class DDM(BaseSPCError):
    """DDM (Drift detection method) [gama2004learning]_ detector.

    :References:

    .. [gama2004learning] Gama, Joao, et al.
        "Learning with drift detection."
        Advances in Artificial Intelligence–SBIA 2004: 17th Brazilian Symposium on
        Artificial Intelligence, Sao Luis, Maranhao, Brazil, September 29-Ocotber 1,
        2004. Proceedings 17. Springer Berlin Heidelberg, 2004.
    """

    config_type = DDMConfig

    def _update(self, value: Union[int, float], **kwargs) -> None:
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
