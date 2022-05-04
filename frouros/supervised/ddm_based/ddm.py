"""DDM (Drift detection method) module."""

from typing import Callable, Dict, Optional, Union, Tuple  # noqa: TYP001

from sklearn.base import BaseEstimator  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore
import numpy as np  # type: ignore

from frouros.supervised.ddm_based.base import DDMBaseConfig, DDMBasedEstimator


class DDMConfig(DDMBaseConfig):
    """DDM (Drift detection method) configuration class."""

    def __init__(
        self,
        warning_level: float = 2.0,
        drift_level: float = 3.0,
        min_num_instances: int = 30,
    ) -> None:
        """Init method.

        :param warning_level: warning level factor
        :type warning_level: float
        :param drift_level: drift level factor
        :type drift_level: float
        :param min_num_instances: minimum numbers of instances
        to start looking for changes
        :type min_num_instances: int
        """
        super().__init__(min_num_instances=min_num_instances)
        self.warning_level = warning_level
        self.drift_level = drift_level

    @property
    def drift_level(self) -> float:
        """Drift level property.

        :return: drift level to use in detecting drift
        :rtype: float
        """
        return self._drift_level  # type: ignore

    @drift_level.setter
    def drift_level(self, value: float) -> None:
        """Drift level setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value <= 0.0:
            raise ValueError("drift level must be greater than 0.0.")
        if value <= self.warning_level:
            raise ValueError("drift level must be greater than warning level.")
        self._drift_level = value

    @property
    def warning_level(self) -> float:
        """Warning level property.

        :return: warning level to use in detecting drift
        :rtype: float
        """
        return self._warning_level

    @warning_level.setter
    def warning_level(self, value: float) -> None:
        """Warning level setter.

        :param value: value to be set
        :type value: float
        :raises ValueError: Value error exception
        """
        if value <= 0.0:
            raise ValueError("warning level must be great than 0.0.")
        self._warning_level = value


class DDM(DDMBasedEstimator):
    """DDM (Drift detection method) algorithm class."""

    def __init__(
        self,
        estimator: BaseEstimator,
        error_scorer: Callable,
        config: DDMConfig,
    ) -> None:
        """Init method.

        :param estimator: sklearn estimator
        :type estimator: BaseEstimator
        :param error_scorer: error scorer function
        :type error_scorer: Callable
        :param config: configuration parameters
        :type config: DDMConfig
        """
        super().__init__(estimator=estimator, config=config)
        self.error_scorer = error_scorer
        self.min_error_rate = float("inf")
        self.min_std = float("inf")

    @property
    def min_error_rate(self) -> float:
        """Minimum error rate property.

        :return: minimum error rate to use
        :rtype: float
        """
        return self._min_error_rate

    @min_error_rate.setter
    def min_error_rate(self, value: float) -> None:
        """Minimum error rate setter.

        :param value: value to be set
        :type value: float
        """
        if value < 0:
            raise ValueError("min_error_rate must be great or equal than 0.")
        self._min_error_rate = value

    @property
    def min_error_rate_plus_std(self) -> float:
        """Minimum error rate + std property.

        :return: minimum error rate + std to determine if a change is happening
        :rtype: float
        """
        return self.min_error_rate + self.min_std

    @property
    def min_std(self) -> float:
        """Minimum standard deviation property.

        :return: minimum standard deviation to use
        :rtype: float
        """
        return self._min_std

    @min_std.setter
    def min_std(self, value: float) -> None:
        """Minimum standard deviation setter.

        :param value: value to be set
        :type value: float
        """
        if value < 0:
            raise ValueError("min_std must be great or equal than 0.")
        self._min_std = value

    def _calculate_error_rate_plus_std(self) -> Tuple[float, float, float]:
        error_rate = self.error_scorer(
            y_true=self.ground_truth, y_pred=self.predictions
        )
        std = np.sqrt(error_rate * (1 - error_rate) / self.num_instances)
        error_rate_plus_std = error_rate + std
        return error_rate_plus_std, error_rate, std

    @staticmethod
    def _check_threshold(
        error_rate_plus_std: float, min_error_rate: float, min_std: float, level: float
    ) -> bool:
        return error_rate_plus_std > min_error_rate + level * min_std

    def _reset(self) -> None:
        super()._reset()
        self.min_error_rate = float("inf")
        self.min_std = float("inf")

    def update(self, y: np.ndarray) -> Dict[str, Optional[Union[float, bool]]]:
        """Update drift detector.

        :param y: input data
        :type y: numpy.ndarray
        :return predicted values
        :rtype: numpy.ndarray
        """
        check_is_fitted(self.estimator)
        X, y_pred = self.delayed_predictions.popleft()  # noqa: N806
        self.num_instances += y_pred.shape[0]

        if self._drift_insufficient_samples:
            drift_completed_flag = self._check_drift_insufficient_samples(X=X, y=y)
            if drift_completed_flag:
                response = self._get_update_response(drift=True, warning=False)
                return response

        self.ground_truth.extend(y.tolist())
        self.predictions.extend(y_pred.tolist())
        if self.num_instances > self.config.min_num_instances:
            error_rate_plus_std, error_rate, std = self._calculate_error_rate_plus_std()

            if error_rate_plus_std < self.min_error_rate_plus_std:
                self.min_error_rate = error_rate
                self.min_std = std

            drift = self._check_threshold(
                error_rate_plus_std=error_rate_plus_std,
                min_error_rate=self.min_error_rate,
                min_std=self.min_std,
                level=self.config.drift_level,  # type: ignore
            )

            if drift:
                # Out-of-Control
                self._drift_case(X=X, y=y)
                warning = True
            else:
                warning = self._check_threshold(
                    error_rate_plus_std=error_rate_plus_std,
                    min_error_rate=self.min_error_rate,
                    min_std=self.min_std,
                    level=self.config.warning_level,  # type: ignore
                )

                if warning:
                    # Warning
                    self._warning_case(X, y)
                else:
                    # In-Control
                    self._normal_case(y)
        else:
            error_rate_plus_std, drift, warning = 0.0, False, False  # type: ignore

        response = self._get_update_response(
            drift=drift, warning=warning, error_rate_plus_std=error_rate_plus_std
        )
        return response
