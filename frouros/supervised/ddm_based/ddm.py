"""DDM (Drift detection method) module."""

from typing import Dict, Optional, Union  # noqa: TYP001

import numpy as np  # type: ignore

from frouros.supervised.ddm_based.base import DDMBaseConfig, DDMErrorBasedEstimator


class DDMConfig(DDMBaseConfig):
    """DDM (Drift detection method) configuration class."""


class DDM(DDMErrorBasedEstimator):
    """DDM (Drift detection method) algorithm class."""

    def update(
        self, y: np.ndarray
    ) -> Dict[str, Optional[Union[float, bool, Dict[str, float]]]]:
        """Update drift detector.

        :param y: input data
        :type y: numpy.ndarray
        :return response message
        :rtype: Dict[str, Optional[Union[float, bool, Dict[str, float]]]]
        """
        X, y_pred, metrics = self._prepare_update(y=y)  # noqa: N806

        if self._drift_insufficient_samples and self._check_drift_insufficient_samples(
            X=X, y=y
        ):
            response = self._get_update_response(
                drift=True, warning=True, metrics=metrics
            )
            return response  # type: ignore

        error_rate_sample = self.error_scorer(y_true=y, y_pred=y_pred)
        self.error_rate += (error_rate_sample - self.error_rate) / self.num_instances

        if self.num_instances >= self.config.min_num_instances:
            error_rate_plus_std, std = self._calculate_error_rate_plus_std()

            self._check_min_values(error_rate_plus_std=error_rate_plus_std, std=std)

            drift_flag = self._check_threshold(
                error_rate_plus_std=error_rate_plus_std,
                min_error_rate=self.min_error_rate,
                min_std=self.min_std,
                level=self.config.drift_level,  # type: ignore
            )

            if drift_flag:
                # Out-of-Control
                self._drift_case(X=X, y=y)
                self.drift = True
                self.warning = True
            else:
                warning_flag = self._check_threshold(
                    error_rate_plus_std=error_rate_plus_std,
                    min_error_rate=self.min_error_rate,
                    min_std=self.min_std,
                    level=self.config.warning_level,  # type: ignore
                )
                if warning_flag:
                    # Warning
                    self._warning_case(X=X, y=y)
                    self.warning = True
                else:
                    # In-Control
                    self._normal_case(X=X, y=y)
                    self.warning = False
                self.drift = False
        else:
            error_rate_plus_std, self.drift, self.warning = 0.0, False, False

        response = self._get_update_response(
            drift=self.drift,
            warning=self.warning,
            error_rate_plus_std=error_rate_plus_std,
            metrics=metrics,
        )
        return response
