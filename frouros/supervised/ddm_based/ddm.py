"""DDM (Drift detection method) module."""

from typing import Dict, Optional, Union  # noqa: TYP001

import numpy as np  # type: ignore

from frouros.supervised.ddm_based.base import DDMBaseConfig, DDMErrorBasedEstimator


class DDMConfig(DDMBaseConfig):
    """DDM (Drift detection method) configuration class."""


class DDM(DDMErrorBasedEstimator):
    """DDM (Drift detection method) algorithm class."""

    def update(
        self,
        y: np.ndarray,
        X: np.ndarray = None,  # noqa: N803
    ) -> Dict[str, Optional[Union[float, bool, Dict[str, float]]]]:
        """Update drift detector.

        :param y: input data
        :type y: numpy.ndarray
        :param X: feature data
        :type X: Optional[numpy.ndarray]
        :return: response message
        :rtype: Dict[str, Optional[Union[float, bool, Dict[str, float]]]]
        """
        X, y_pred, metrics = self._prepare_update(y=y)  # noqa: N806

        if self._drift_insufficient_samples:
            self._insufficient_samples_case(X=X, y=y)
            if not self._check_drift_sufficient_samples:
                # Drift has been detected but there are no enough samples
                # to train a new model from scratch
                return self._insufficient_samples_response(metrics=metrics)
            # There are enough samples to train a new model from scratch
            self._complete_delayed_drift()

        error_rate = self.error_scorer(y_true=y, y_pred=y_pred)
        self.error_rate.update(value=error_rate)

        specific_attributes = self._get_specific_response_attributes()

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
                self._drift_case(X=X, y=y)
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
                    self._warning_case(X=X, y=y)
                    self.warning = True
                else:
                    # In-Control
                    self._normal_case(X=X, y=y)
                    self.warning = False
                self.drift = False
        else:
            self._normal_case(X=X, y=y)
            self.drift, self.warning = False, False

        return self._update_response(
            specific_attributes=specific_attributes, metrics=metrics
        )
