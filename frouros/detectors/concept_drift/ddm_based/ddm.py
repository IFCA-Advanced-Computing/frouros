"""DDM (Drift detection method) module."""

from contextlib import suppress
from typing import Union

from frouros.detectors.concept_drift.ddm_based.base import DDMBaseConfig, DDMErrorBased


class DDMConfig(DDMBaseConfig):
    """DDM (Drift detection method) configuration class."""


class DDM(DDMErrorBased):
    """DDM (Drift detection method) algorithm class."""

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
